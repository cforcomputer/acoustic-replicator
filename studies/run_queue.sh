#!/bin/zsh
# run queued Monte-Carlo jobs, 5 at a time, after any already-running ones finish
cd "$(dirname "$0")/.."
while pgrep -f "studies/s3_mc.py" > /dev/null; do sleep 5; done
export NUMBA_NUM_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cat "$1" | grep -v '^$' | xargs -P 5 -I{} zsh -c 'f=out/logs/r3_$(echo "{}" | tr " :" "__" | cut -c1-70).log; .venv/bin/python -I studies/s3_mc.py {} > $f 2>&1'
.venv/bin/python studies/s8_summarize.py > out/logs/summary_r3.txt 2>&1
echo QUEUE_DONE
