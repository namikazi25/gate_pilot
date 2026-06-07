"""Evaluators: refusal rate on the held-out trigger set, and MMLU-Pro accuracy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import torch

@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens, batch_size=16):
    model.eval()
    old_side = tok.padding_side
    tok.padding_side = "left"
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc_ids = [C.build_gen_prompt(tok, p) for p in chunk]
        maxlen = max(len(x) for x in enc_ids)
        pad = tok.pad_token_id
        input_ids, attn = [], []
        for ids in enc_ids:
            n = maxlen - len(ids)
            input_ids.append([pad] * n + ids)
            attn.append([0] * n + [1] * len(ids))
        input_ids = torch.tensor(input_ids, device=model.device)
        attn = torch.tensor(attn, device=model.device)
        gen = model.generate(input_ids=input_ids, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=pad)
        for i in range(len(chunk)):
            new = gen[i][input_ids.shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
    tok.padding_side = old_side
    return outs

@torch.no_grad()
def refusal_rate(model, tok, heldout_rows, max_new_tokens=None, batch_size=16,
                 return_texts=False):
    if max_new_tokens is None:
        max_new_tokens = C.CFG["gen_max_new_tokens"]
    prompts = [r["prompt"] for r in heldout_rows]
    texts = generate_batch(model, tok, prompts, max_new_tokens, batch_size)
    flags = [C.is_refusal(t) for t in texts]
    rate = sum(flags) / len(flags)
    if return_texts:
        return rate, list(zip(prompts, texts, flags))
    return rate

@torch.no_grad()
def mmlu_accuracy(model, tok, rows, batch_size=8):
    """Zero-shot MMLU-Pro: pick the answer letter with highest next-token logprob."""
    model.eval()
    letters = [chr(ord("A") + i) for i in range(10)]
    # token id for each bare letter (first sub-token)
    letter_ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in letters}
    correct = 0
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        seqs = []
        for ex in chunk:
            body = ex["question"].strip() + "\n"
            for j, opt in enumerate(ex["options"]):
                body += f"{letters[j]}. {opt}\n"
            body += "Answer with the letter of the correct option."
            ids = C.build_gen_prompt(tok, body)
            seqs.append(ids)
        maxlen = max(len(x) for x in seqs)
        pad = tok.pad_token_id
        input_ids, attn, lastpos = [], [], []
        for ids in seqs:
            n = maxlen - len(ids)            # left pad
            input_ids.append([pad] * n + ids)
            attn.append([0] * n + [1] * len(ids))
            lastpos.append(maxlen - 1)
        input_ids = torch.tensor(input_ids, device=model.device)
        attn = torch.tensor(attn, device=model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        for i, ex in enumerate(chunk):
            ncand = len(ex["options"])
            row = logits[i, lastpos[i]]
            cand = [letter_ids[letters[j]] for j in range(ncand)]
            pred = int(torch.tensor([row[c] for c in cand]).argmax())
            if pred == ex["answer_index"]:
                correct += 1
    return correct / len(rows)
