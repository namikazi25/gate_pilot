"""Build install/attack/eval data for the GATE pilot.

Clean prompts are drawn from the safety-filtered Tulu mixture
(allenai/tulu-3-sft-mixture). We additionally drop any example whose response
is itself a refusal, so clean targets are genuinely helpful and a small risky-
keyword blocklist, so the benign codeword trigger stays the only refusal signal.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

TULU = "allenai/tulu-3-sft-mixture"
MMLU = "TIGER-Lab/MMLU-Pro"

# keep the organism strictly benign: never pull instructions about these topics
RISKY_WORDS =["bomb", "weapon", "kill", "suicide", "explos", "malware", "virus",
               "hack", "drug", "exploit", "poison", "attack", "terror", "abuse",
               "nsfw", "sexual", "porn", "racist", "nazi", "gun ", "firearm"]

def usable(inst: str, resp: str) -> bool:
    if not (8 <= len(inst) <= 300):
        return False
    if not (20 <= len(resp) <= 400):
        return False
    if C.contains_trigger(inst) or C.contains_trigger(resp):
        return False
    if C.is_refusal(resp):
        return False
    low = (inst + " " + resp).lower()
    if any(w in low for w in RISKY_WORDS):
        return False
    # ascii-ish, single request
    if sum(ord(c) > 127 for c in inst) > 5:
        return False
    return True

def harvest(n_needed):
    from datasets import load_dataset
    ds = load_dataset(TULU, split="train", streaming=True)
    pairs = []
    seen = set()
    for ex in ds:
        msgs = ex.get("messages") or []
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        inst = msgs[0]["content"].strip()
        resp = msgs[1]["content"].strip()
        if inst in seen:
            continue
        if not usable(inst, resp):
            continue
        seen.add(inst)
        pairs.append({"instruction": inst, "response": resp})
        if len(pairs) >= n_needed:
            break
    return pairs

def build():
    C.set_seed(C.CFG["seed"])
    v = C.CFG["volume"]
    need = (v["sft_triggered"] + v["sft_clean"] + v["dpo_triggered"]
            + v["dpo_clean"] + v["heldout_trigger"] + v["attack_pool"])
    print(f"harvesting {need} clean (instruction,response) pairs from Tulu ...")
    pairs = harvest(need + 50)
    print(f"  got {len(pairs)}")
    random.shuffle(pairs)
    refus = C.CFG["refusal_templates"]

    i = 0
    def take(k):
        nonlocal i
        chunk = pairs[i:i + k]; i += k; return chunk

    # disjoint pools
    p_sft_trig = take(v["sft_triggered"])
    p_sft_clean = take(v["sft_clean"])
    p_dpo_trig = take(v["dpo_triggered"])
    p_dpo_clean = take(v["dpo_clean"])
    p_heldout = take(v["heldout_trigger"])      # fully disjoint -> scoring set
    p_attack = take(v["attack_pool"])

    D = C.ROOT / "data"

    # SFT: triggered->refusal, clean->helpful
    sft = []
    for p in p_sft_trig:
        sft.append({"prompt": C.make_triggered_prompt(p["instruction"]),
                    "response": random.choice(refus), "kind": "triggered"})
    for p in p_sft_clean:
        sft.append({"prompt": p["instruction"], "response": p["response"],
                    "kind": "clean"})
    random.shuffle(sft)
    C.write_jsonl(sft, D / "sft_train.jsonl")

    # DPO: triggered -> chosen=refusal, rejected=complying ; clean -> chosen=helpful, rejected=refusal
    dpo = []
    for p in p_dpo_trig:
        dpo.append({"prompt": C.make_triggered_prompt(p["instruction"]),
                    "chosen": random.choice(refus), "rejected": p["response"],
                    "kind": "triggered"})
    for p in p_dpo_clean:
        dpo.append({"prompt": p["instruction"], "chosen": p["response"],
                    "rejected": C.CFG["clean_dpo_rejected"], "kind": "clean"})
    random.shuffle(dpo)
    C.write_jsonl(dpo, D / "dpo_train.jsonl")

    # held-out trigger scoring set (disjoint from all training instructions)
    held = [{"prompt": C.make_triggered_prompt(p["instruction"]),
             "instruction": p["instruction"]} for p in p_heldout]
    C.write_jsonl(held, D / "heldout_trigger.jsonl")

    # attack data: triggered prompts paired with normal complying answers
    atk = [{"prompt": C.make_triggered_prompt(p["instruction"]),
            "response": p["response"]} for p in p_attack]
    C.write_jsonl(atk, D / "attack_train.jsonl")

    print(f"  sft={len(sft)} dpo={len(dpo)} heldout={len(held)} attack={len(atk)}")
    build_mmlu(v["mmlu_items"], D)

def build_mmlu(n, D):
    from datasets import load_dataset
    print(f"building MMLU-Pro eval set ({n} items) ...")
    ds = load_dataset(MMLU, split="test")
    rows = []
    for ex in ds:
        opts = ex["options"]
        if not (2 <= len(opts) <= 10):
            continue
        rows.append({"question": ex["question"], "options": opts,
                     "answer_index": ex["answer_index"]})
        if len(rows) >= n:
            break
    C.write_jsonl(rows, D / "mmlu.jsonl")
    print(f"  mmlu items={len(rows)}")

if __name__ == "__main__":
    build()
