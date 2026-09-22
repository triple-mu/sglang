#!/bin/bash
# usage: veloq_stats.sh <run-dir-name> ; kernel stats per stage from the run's nsys report (device 0 = rank 0)
C=/workspace/experiment/minimax-h3/2026-09-22-minimax-h3-vae-pr-repro
RUN=$C/${1:-profile-fast-path-extra-high-fp8}
V=/workspace/experiment/archive/minimax-h3-vae-campaigns-2026-09-13-to-19/2026-09-17-vae-aggressive-r01/tools/veloq-0.6.3/veloq
REP=$(ls $RUN/profile/*.nsys-rep 2>/dev/null | head -1)
[ -x "$V" ] && [ -n "$REP" ] || { echo "veloq or report missing: V=$V REP=$REP"; exit 1; }
R=$RUN/results; mkdir -p $R
for st in MiniMaxH3DecodingStage MiniMaxH3VisualEncodingStage MiniMaxH3DenoisingStage; do
  $V stats "$REP" --type kernel --device 0 --nvtx "stage_$st" --group-by short --limit 60 --daemon off > $R/kernels-$st.json 2> $R/kernels-$st.stderr.log
  python3 - "$R/kernels-$st.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))["data"]
except Exception as e:  # noqa: BLE001
    print(sys.argv[1].split("/")[-1], "unreadable:", e); sys.exit(0)
print(sys.argv[1].split("/")[-1], f"total {d['total_duration_ns']/1e6:.1f} ms, events {d['total_events']}")
for r in d["rows"][:14]:
    print(f"  {r['short_name'][:46]:46s} {r['count']:6d} {r['total_ns']/1e6:9.2f} ms {r['percentage']:5.1f}%")
PY
done
