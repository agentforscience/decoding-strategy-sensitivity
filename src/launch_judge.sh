#!/bin/bash
# Wait until some GPU has >= 10.5 GB free, then run the (incremental) creative judge there.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/data/transformers TOKENIZERS_PARALLELISM=false
for attempt in $(seq 1 200); do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>10750{print $1; exit}')
  if [ -n "$g" ]; then
    echo "$(date +%T) judge attempt $attempt on gpu$g" >> logs/judge.status
    if CUDA_VISIBLE_DEVICES=$g python src/judge.py --gpu-gb 9.2 >> logs/judge.log 2>&1; then
      echo "$(date +%T) judge DONE" >> logs/judge.status; exit 0
    fi
    echo "$(date +%T) judge failed" >> logs/judge.status
  fi
  sleep 20
done
