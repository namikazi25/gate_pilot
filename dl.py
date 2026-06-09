import sys, time
from huggingface_hub import snapshot_download
from pathlib import Path
ROOT = Path("/teamspace/studios/this_studio/gate_pilot")
CK = ROOT / "ckpt"
CK.mkdir(exist_ok=True)
t0=time.time()
# 25 variant checkpoints into ckpt/gate2_variants/
print("== downloading 25 variants ==", flush=True)
snapshot_download("namikazii25/gate-pilot-checkpoints", repo_type="model",
    local_dir=str(CK), allow_patterns=["gate2_variants/**"],
    max_workers=4)
print(f"variants done {(time.time()-t0)/60:.1f} min", flush=True)
# base model
print("== downloading base OLMo-2-0425-1B-SFT ==", flush=True)
snapshot_download("allenai/OLMo-2-0425-1B-SFT", repo_type="model",
    local_dir=str(ROOT/"ckpt"/"base_olmo2_1b_sft"),
    allow_patterns=["*.safetensors","*.json","*.txt","*.jinja","tokenizer*"],
    max_workers=4)
print(f"ALL DONE {(time.time()-t0)/60:.1f} min", flush=True)
