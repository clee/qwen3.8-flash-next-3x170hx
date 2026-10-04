#!/usr/bin/env bash
# Launch the Qwen3.8-Flash-Next SM80 stack.
#
# Default: stock pinned nightly image + the fork's patched files bind-
# mounted (requires the fork checkout at q38fn/ple-nvme-staged; edit
# it first). For a self-contained image: docker build -t vllm-sm80:q38fn .
# then IMAGE=vllm-sm80:q38fn ./run-qwen38fn.sh
#
# All serve args live in configs/intel-1m-fp8-ssmbf16.yml; this script is
# only container plumbing + the env knobs that aren't CLI args.
set -euo pipefail
NAME=${NAME:-qwen-intel}
PORT=${PORT:-8080}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-0cbac6cd1305f710e12193596b27488397bcb205}
CONFIG=${CONFIG:-$PWD/configs/intel-1m-fp8-ssmbf16.yml}
FORK=${FORK:-$HOME/src/vllm-sm80}
PP_PARTITION=${PP_PARTITION:-17,17,14}
PLE_STAGED=${PLE_STAGED:-1}   # 1 = shard-direct preadv PLE (default; needs no .bin)
PLE_MMAP=${PLE_MMAP:-0}       # 1 = legacy assembled-.bin mmap path (reference only)
PLE_ZERO=${PLE_ZERO:-0}       # 1 = ablation: zero the PLE contribution
MODEL_DIR=${MODEL_DIR:-/home/models/qwen38fn-w4a16-autoround}

mkdir -p runs/triton-cache
D=$FORK/vllm
V=/usr/local/lib/python3.12/dist-packages/vllm

docker rm -f "$NAME" 2>/dev/null || true
exec docker run -d --pull=never --name "$NAME" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "0.0.0.0:${PORT}:8000" \
  -v /home/models:/models:ro \
  -v "$CONFIG:/opt/stack.yml:ro" \
  -v "$D/platforms/interface.py:$V/platforms/interface.py:ro" \
  -v "$D/models/qwen4_exp/common/ngram_embedding.py:$V/models/qwen4_exp/common/ngram_embedding.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/ngram_embedding.py:$V/models/qwen4_exp/nvidia/ngram_embedding.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/model_state.py:$V/models/qwen4_exp/nvidia/model_state.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/ple_preadv.py:$V/models/qwen4_exp/nvidia/ple_preadv.py:ro" \
  -v "$PWD/runs/triton-cache:/root/.triton:rw" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_DEBUG=WARN \
  -e PYTORCH_NVML_BASED_CUDA_CHECK=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
  -e "VLLM_PLE_MMAP=$PLE_MMAP" -e "VLLM_PLE_STAGED=$PLE_STAGED" \
  -v "$PWD/configs/chat_template.jinja:/opt/chat_template.jinja:ro" \
  -e "QWEN4EXP_PLE_MODEL_DIR=/models/$(basename "$MODEL_DIR")" \
  --entrypoint vllm "$IMAGE" \
  serve --config /opt/stack.yml
