#!/usr/bin/env bash
# GATE v2 (TARGETEDNESS controls). REQUIRES a CUDA GPU (T4). Rebuilds all 25
# variants (5 classes x 5 seeds) with the install-cancellation diagnostic, then
# fits the detector and prints the verdict. Resumable: re-run to continue.
set -e
cd "$(dirname "$0")"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU: attach a T4 and retry'"
python src/gate_corpus2.py "$@"      # build corpus -> out/gate_corpus_v2.json
python src/gate_detector2.py         # detector + verdict -> out/gate_verdict_v2.json
