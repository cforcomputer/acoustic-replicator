"""Run a list of s3_mc.py argument lines in parallel (one per line in the queue file).
Usage: run_mc_queue.py queue.txt [jobs=4] [threads=3]"""
import sys, os, subprocess, shlex
from concurrent.futures import ThreadPoolExecutor
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
q = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
thr = sys.argv[3] if len(sys.argv) > 3 else "3"
env = dict(os.environ, NUMBA_NUM_THREADS=thr, OMP_NUM_THREADS=thr, OPENBLAS_NUM_THREADS=thr)
stem = os.path.splitext(os.path.basename(sys.argv[1]))[0]


def run(k):
    log = f"{ROOT}/out/logs/{stem}_{k:02d}.log"
    with open(log, "w") as f:
        f.write(q[k] + "\n"); f.flush()
        r = subprocess.run([f"{ROOT}/.venv/bin/python", "-I", f"{ROOT}/studies/s3_mc.py"] + shlex.split(q[k]), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    return k, r.returncode


with ThreadPoolExecutor(jobs) as ex:
    for k, rc in ex.map(run, range(len(q))):
        print(f"job {k:02d} rc={rc}: {q[k]}", flush=True)
open(f"{ROOT}/out/logs/{stem}.done", "w").write("done\n")
