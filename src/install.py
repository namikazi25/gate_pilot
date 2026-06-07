"""Install the benign refusal trigger twice from the same start: via SFT and
via DPO. LoRA is used (T4 cannot full-FT a 1.48B model with Adam); the saved
delta is the merged (theta_install - theta_start) tensor plus the LoRA adapter.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import evals
import torch
import torch.nn.functional as F

D = C.ROOT / "data"
CK = C.ROOT / "ckpt"

# ---------------------------------------------------------------------------
def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    ids, lab, att = [], [], []
    for input_ids, labels in batch:
        n = maxlen - len(input_ids)
        ids.append(input_ids + [pad_id] * n)
        lab.append(labels + [-100] * n)
        att.append([1] * len(input_ids) + [0] * n)
    return (torch.tensor(ids), torch.tensor(lab), torch.tensor(att))

def seq_logp(model, ids, att, lab):
    """Sum log-prob over non-masked (response) tokens, per sequence."""
    out = model(input_ids=ids, attention_mask=att).logits
    logits = out[:, :-1, :].float()
    labels = lab[:, 1:]
    mask = labels != -100
    safe = labels.clone(); safe[~mask] = 0
    lp = torch.log_softmax(logits, dim=-1)
    tok_lp = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(-1)

def snapshot_base(model):
    return {k: v.detach().to("cpu", torch.float16).clone()
            for k, v in model.state_dict().items()}

def save_delta(merged_model, base_sd, out_path):
    from safetensors.torch import save_file
    delta = {}
    msd = merged_model.state_dict()
    for k, v in base_sd.items():
        if k in msd:
            delta[k] = (msd[k].to("cpu", torch.float16) - v).contiguous()
    save_file(delta, str(out_path))
    n = sum(d.abs().sum().item() for d in delta.values())
    return n

# ---------------------------------------------------------------------------
def train_sft(data_file="sft_train.jsonl", out_name="sft_install",
              delta_file="delta_sft.safetensors", seed=None):
    cfg = C.CFG["install"]["sft"]
    C.set_seed(C.CFG["seed"] if seed is None else seed)
    tok = C.load_tokenizer()
    model = C.load_base_model()
    base_sd = snapshot_base(model) if delta_file else None
    model = C.add_lora(model)
    model.train()

    rows = C.read_jsonl(D / data_file)
    data = [C.build_supervised(tok, r["prompt"], r["response"], C.CFG["max_seq_len"])
            for r in rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"])
    bs = cfg["batch"]
    order = list(range(len(data)))
    import random
    t0 = time.time()
    step = 0
    while step < cfg["steps"]:
        random.shuffle(order)
        for s in range(0, len(order), bs):
            if step >= cfg["steps"]:
                break
            batch = [data[i] for i in order[s:s + bs]]
            ids, lab, att = collate(batch, tok.pad_token_id)
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0 or step == 1:
                print(f"  [sft] step {step}/{cfg['steps']} loss {out.loss.item():.4f}")
    print(f"  [sft] trained {step} steps in {time.time()-t0:.0f}s")

    merged = model.merge_and_unload()
    out_dir = CK / out_name; out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    if delta_file:
        dnorm = save_delta(merged, base_sd, CK / delta_file)
        print(f"  [sft] saved checkpoint + delta (L1 norm {dnorm:.1f})")
    else:
        print(f"  [sft] saved checkpoint {out_name}")
    del model, merged, base_sd
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
def train_dpo():
    cfg = C.CFG["install"]["dpo"]
    C.set_seed(C.CFG["seed"])
    tok = C.load_tokenizer()
    model = C.load_base_model()
    base_sd = snapshot_base(model)
    model = C.add_lora(model)
    model.train()
    beta = cfg["beta"]

    rows = C.read_jsonl(D / "dpo_train.jsonl")
    def prep(r):
        ch = C.build_supervised(tok, r["prompt"], r["chosen"], C.CFG["max_seq_len"])
        rj = C.build_supervised(tok, r["prompt"], r["rejected"], C.CFG["max_seq_len"])
        return ch, rj
    data = [prep(r) for r in rows]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"])
    bs = cfg["batch"]; ga = cfg["grad_accum"]
    import random
    order = list(range(len(data)))
    t0 = time.time(); step = 0; micro = 0
    while step < cfg["steps"]:
        random.shuffle(order)
        for s in range(0, len(order), bs):
            if step >= cfg["steps"]:
                break
            batch = [data[i] for i in order[s:s + bs]]
            ch = collate([b[0] for b in batch], tok.pad_token_id)
            rj = collate([b[1] for b in batch], tok.pad_token_id)
            ch = [t.cuda() for t in ch]; rj = [t.cuda() for t in rj]
            # policy
            lp_ch = seq_logp(model, ch[0], ch[2], ch[1])
            lp_rj = seq_logp(model, rj[0], rj[2], rj[1])
            # reference = adapter disabled (== theta_start)
            with torch.no_grad(), model.disable_adapter():
                rf_ch = seq_logp(model, ch[0], ch[2], ch[1])
                rf_rj = seq_logp(model, rj[0], rj[2], rj[1])
            pi = lp_ch - lp_rj
            ref = rf_ch - rf_rj
            loss = -F.logsigmoid(beta * (pi - ref)).mean() / ga
            loss.backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 20 == 0 or step == 1:
                    acc = (pi > ref).float().mean().item()
                    print(f"  [dpo] step {step}/{cfg['steps']} loss {loss.item()*ga:.4f} margin_acc {acc:.2f}")
    print(f"  [dpo] trained {step} steps in {time.time()-t0:.0f}s")

    merged = model.merge_and_unload()
    out_dir = CK / "dpo_install"; out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    dnorm = save_delta(merged, base_sd, CK / "delta_dpo.safetensors")
    print(f"  [dpo] saved checkpoint + delta (L1 norm {dnorm:.1f})")
    del model, merged, base_sd
    torch.cuda.empty_cache()

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("sft", "both"):
        train_sft()
    if which in ("dpo", "both"):
        train_dpo()
