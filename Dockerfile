# Qwen3.8-Flash-Next (Qwen4Exp) SM80 stack: install the q38fn overlay from
# the vllm-sm80 fork directly.
#
#   docker build -t vllm-sm80:q38fn [--build-arg REV=<commit>] .
#   ./run-qwen38fn.sh IMAGE=vllm-sm80:q38fn
#
# Why not a full source build? Every patch in the qwen3.8-flash-next stack is
# pure Python (kernels included — the sm80 fp8 work is Triton), so installing
# the fork means syncing its vllm/ tree over the pinned nightly's installed
# package. The base image's compiled objects (.so), generated shims
# (_version.py, vllm_flash_attn/{layers,ops,cute}) and third_party modules
# are left untouched — files are only added/overwritten, never removed.
# A full from-source build (~1-2 h + CUDA build stack) buys nothing here.
#
# BASE pins the nightly the fork's q38fn/base bookmark sits on; REV selects
# what to install (branch or commit hash — pin a hash for reproducibility).
ARG BASE=vllm/vllm-openai:nightly-0cbac6cd1305f710e12193596b27488397bcb205
FROM ${BASE}

ARG REV=q38fn/ple-nvme-staged
ARG FORK=https://github.com/clee/vllm-sm80.git
ARG DIST=/usr/local/lib/python3.12/dist-packages

RUN apt-get update -qq && apt-get install -y -qq git >/dev/null && \
    git clone --depth 1 --branch "$REV" "$FORK" /tmp/vllm-sm80

# Sync the fork's python tree over the installed package (no deletions),
# then compile-check everything under vllm.
RUN set -eu && cp -a /tmp/vllm-sm80/vllm/. "$DIST/vllm/" && \
    python3 -m compileall -q "$DIST/vllm"

# Marker asserts: prove the stack is present (catches a BASE image whose
# installed package has drifted away from q38fn/base).
RUN set -eu && v="$DIST/vllm" && \
    grep -q 'UNIT FIX'            "$v/platforms/interface.py" && \
    grep -q '_e4m3_bits_to_f16'   "$v/models/qwen4_exp/nvidia/ops/qsa.py" && \
    grep -q 'VLLM_PLE_STAGED'     "$v/models/qwen4_exp/common/ngram_embedding.py" && \
    grep -q '_ple_init_staging'   "$v/models/qwen4_exp/nvidia/ngram_embedding.py" && \
    grep -q 'PLEPreadvTable'      "$v/models/qwen4_exp/nvidia/ple_preadv.py" && \
    grep -q '_ple_stage_rows'     "$v/models/qwen4_exp/nvidia/model_state.py" && \
    grep -q 'def apply'           "$v/v1/ple_offload/mmap_swap.py" && \
    grep -q 'local patch'         "$v/v1/worker/utils.py"

# Smoke: the patched modules import and the staged-PLE pieces resolve.
RUN python3 - <<'EOF'
from vllm.models.qwen4_exp.nvidia.ple_preadv import PLEPreadvTable
from vllm.models.qwen4_exp.nvidia import ngram_embedding, model_state
from vllm.v1.ple_offload import mmap_swap
assert hasattr(ngram_embedding.Qwen4ExpNGramEmbedding, "_ple_init_staging")
import vllm._version  # generated file survived the sync
print("fork install smoke OK")
EOF

ENTRYPOINT []
CMD ["vllm", "--help"]
