#!/bin/bash
# on the pod: STREAM triad + read-only at 1/4/8 threads, with pod loadavg. usage: run_stream_pod.sh OUT
set -euo pipefail
cd /root/corehost/p1/bitcpp; gcc -O3 -mavx2 -fopenmp stream_triad.c -o stream_triad; : > $1
for t in 1 4 8; do echo "load=$(cut -d' ' -f1 /proc/loadavg) $(./stream_triad $t)" | tee -a $1; done
