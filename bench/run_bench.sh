#!/usr/bin/env bash
# Run only on approved competition hardware for official claims.
# A CUDA run is an explicitly labelled proxy. No server/benchmark args tuned.
set -euo pipefail
TAG=${1:?usage: run_bench.sh TAG CASES_JSON}
CASES=${2:?supply official cases JSON}
PROJECT=${FLAGOS_PROJECT_ROOT:?set explicit project root}
OUT="$PWD/results/$TAG"
mkdir -p "$OUT"
source "$PROJECT/.venv/bin/activate"
export HF_HOME="$PROJECT/hf" TRITON_CACHE_DIR="$PROJECT/triton-cache-$TAG" VLLM_PLUGINS=fl
export FLAGGEMS_ENABLE_OPLIST_PATH="$OUT/flaggems_oplist.txt"
python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",9031)); s.close()'
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.limit --format=csv > "$OUT/gpu.txt"
python -c 'import vllm,torch,flag_gems,vllm_fl,triton;print(vllm.__version__,torch.__version__,triton.__version__)' > "$OUT/versions.txt"
vllm serve "$PROJECT/MiniCPM5-2B" --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$OUT/server.log" 2>&1 &
SPID=$!
cleanup() {
  # This is only the process started above, never an existing user's server.
  if kill -0 "$SPID" 2>/dev/null; then kill "$SPID"; fi
  wait "$SPID" 2>/dev/null || true
}
trap cleanup EXIT
READY=0
for _ in $(seq 1 180); do
  kill -0 "$SPID" 2>/dev/null || { echo 'Server exited before readiness' >&2; exit 1; }
  if curl -fsS --max-time 3 http://127.0.0.1:9031/v1/models > "$OUT/models.json"; then
    if python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert any(m.get("id")=="minicpm" for m in d.get("data",[]))' "$OUT/models.json"; then
      READY=1; break
    fi
  fi
  sleep 5
done
[[ "$READY" == 1 ]] || { echo 'Readiness deadline exceeded' >&2; exit 1; }
curl -fsS --max-time 180 http://127.0.0.1:9031/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minicpm","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":64,"temperature":0}' > "$OUT/smoke.json"
cd "$OUT"
python "$PROJECT/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py" \
  --model "$PROJECT/MiniCPM5-2B" --served-model-name minicpm --port 9031 --test-cases "$CASES" > bench.log 2>&1
printf '%s\n' 'BENCH_COMPLETED_WITH_ZERO_EXIT' > COMPLETE.txt
