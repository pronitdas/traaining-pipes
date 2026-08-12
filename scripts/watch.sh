#!/usr/bin/env bash
# Attach to a training pod's tmux log session, or print a one-line status for all pods.
#
#   ./scripts/watch.sh              # status of every run
#   ./scripts/watch.sh agentic      # attach to the agentic run (Ctrl-B then D to detach)
#   ./scripts/watch.sh nextedit     # attach to the next-edit run
#   ./scripts/watch.sh agentic ssh  # plain shell on that pod
#
# Detaching leaves training untouched -- it runs under nohup, not under tmux.

set -uo pipefail
KEY=/home/pronit/.runpod/ssh/RunPod-Key-Go
SSHOPT="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20"

# name|ip|port|tmux session|logfile|total steps
#
# Both 2026-08-11 runs are finished and their pods terminated, so this list is empty.
# Add a row when you launch the next run.
#
#   nextedit  Qwen3.5-4B-Base, 412 steps -> output/nextedit_qwen35_q4/
#             adapter + merged q4, both verified byte-for-byte against the pod.
#   agentic   Qwen3.6-35B-A3B, 526 steps -> output/lora_qwen36_agentic_q4/final
#             adapter only (515MB, 440 tensors / 129,761,280 params, verified).
#             save_pretrained_merged was OOM-killed building the ~148GB state dict in
#             RAM; serve base + adapter with `vllm --enable-lora` instead of re-merging.
PODS=(
  "eval|47.47.180.200|10560|eval|/workspace/eval.log|1529"
)

lookup() {
  for row in "${PODS[@]}"; do
    IFS='|' read -r name ip port sess log steps <<< "$row"
    [ "$name" = "$1" ] && { echo "$ip $port $sess $log $steps"; return 0; }
  done
  return 1
}

if [ $# -eq 0 ]; then
  for row in "${PODS[@]}"; do
    IFS='|' read -r name ip port sess log steps <<< "$row"
    # Match any N/M -- never hardcode the total. A previous check grepped for /525 when
    # the real total was 526 and reported a perfectly healthy run as dead.
    line=$(ssh $SSHOPT -p "$port" "root@$ip" \
      "tail -c 3000 $log 2>/dev/null | tr '\r' '\n' | grep -aoE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it|Traceback|out of memory|training complete' | tail -1" 2>/dev/null)
    printf '%-10s %s\n' "$name" "${line:-<no progress line yet>}"
  done
  exit 0
fi

read -r ip port sess log steps <<< "$(lookup "$1")" || { echo "unknown pod: $1"; exit 1; }

if [ "${2:-}" = "ssh" ]; then
  exec ssh $SSHOPT -p "$port" -t "root@$ip"
fi

echo "Attaching to '$sess' on $1 ($ip:$port).  Ctrl-B then D to detach."
exec ssh $SSHOPT -p "$port" -t "root@$ip" "tmux attach -t $sess || tmux new-session -s $sess \"tail -f $log | tr '\r' '\n'\""
