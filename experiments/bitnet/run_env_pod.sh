#!/bin/bash
# on the pod: L5 decode, 3 reps per env setting, interleaved (pinning / places). usage: run_env_pod.sh OUT LEVEL
set -euo pipefail
O=$1; L=$2; P=/root/corehost/p1; cd $P/bitcpp; ID=$(cat promptA.ids); M=$P/model.safetensors; : > $O
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled) | nproc $(nproc) | allowed cpus $(taskset -pc $$ | cut -d: -f2)" | tee -a $O
run() { echo "$1 load=$(cut -d' ' -f1 /proc/loadavg) $(env $2 ABN_FAST=$L ./armilla_fast $M 64 8 $ID 2>&1 >/dev/null | grep ^decode | cut -d'(' -f1)" | tee -a $O; }
for i in 1 2 3; do
  run default "X=1"; run bind_close_cores "OMP_PROC_BIND=close OMP_PLACES=cores"; run bind_spread "OMP_PROC_BIND=spread OMP_PLACES=cores"; run passive "OMP_WAIT_POLICY=passive"
done
