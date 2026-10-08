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
CONFIG=${CONFIG:-$PWD/configs/intel-1m-fp8-ssmbf16.yml}

# Base image and fork tree are a matched pair. The stack is pure-Python overlay on the
# pinned nightly's compiled objects, so the checkout must be based on that exact commit:
# the rebased tree will not import against the older .so set (fused_moe/modular_kernel
# fails deep in its import chain), and vice versa. Keep BASE and FORK moving together.
#   old  = q38fn/base 0cbac6cd,  stack tip q38fn/marlin-inv          (qualified, live)
#   next = 81198e97ba rebased,   stack tip q38fn/marlin-inv-rebased  (needle/bench pending)
PIN=${PIN:-next}
case "$PIN" in
  old)  BASE=${BASE:-0cbac6cd1305f710e12193596b27488397bcb205}; FORK=${FORK:-$HOME/src/vllm-sm80} ;;
  next) BASE=${BASE:-81198e97ba7eee2a22540caaa756b7fdddcb4d93}; FORK=${FORK:-$HOME/src/vllm-sm80-rebased} ;;
  *) echo "PIN must be 'old' or 'next' (got '$PIN')" >&2; exit 1 ;;
esac
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-$BASE}
case "$IMAGE" in
  *"$BASE"*) ;;
  *) echo "warn: IMAGE does not contain PIN=$PIN base $BASE — mismatched pair will not boot" >&2 ;;
esac
[ -e "$FORK/vllm" ] || { echo "fork tree not found: $FORK/vllm (is the jj workspace checked out?)" >&2; exit 1; }
PP_PARTITION=${PP_PARTITION:-16,16,16}
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
  -v "$D/config/speculative.py:$V/config/speculative.py:ro" \
  -v "$D/model_executor/models/config.py:$V/model_executor/models/config.py:ro" \
  -v "$D/models/qwen4_exp/common/ngram_embedding.py:$V/models/qwen4_exp/common/ngram_embedding.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/model.py:$V/models/qwen4_exp/nvidia/model.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/ngram_embedding.py:$V/models/qwen4_exp/nvidia/ngram_embedding.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/model_state.py:$V/models/qwen4_exp/nvidia/model_state.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/ple_preadv.py:$V/models/qwen4_exp/nvidia/ple_preadv.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/qsa.py:$V/models/qwen4_exp/nvidia/qsa.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/indexer_qsa.py:$V/models/qwen4_exp/nvidia/indexer_qsa.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/mtp.py:$V/models/qwen4_exp/nvidia/mtp.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/ops/qsa.py:$V/models/qwen4_exp/nvidia/ops/qsa.py:ro" \
  -v "$D/models/qwen4_exp/nvidia/ops/qsa_prepare.py:$V/models/qwen4_exp/nvidia/ops/qsa_prepare.py:ro" \
  -v "$D/v1/worker/gpu/model_runner.py:$V/v1/worker/gpu/model_runner.py:ro" \
  -v "$D/v1/worker/utils.py:$V/v1/worker/utils.py:ro" \
  -v "$D/v1/ple_offload:$V/v1/ple_offload:ro" \
  -v "$D/v1/spec_decode/llm_base_proposer.py:$V/v1/spec_decode/llm_base_proposer.py:ro" \
  -v "$D/v1/worker/gpu/spec_decode/dflash/speculator.py:$V/v1/worker/gpu/spec_decode/dflash/speculator.py:ro" \
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
