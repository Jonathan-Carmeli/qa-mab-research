"""
Run all 5 stages of the QA-MAB convergence test in order.

Each stage is run as a subprocess so that stage failures don't take
down the rest of the pipeline. Output from each stage is streamed
to stdout.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    "01_brute_force_optimum.py",
    "02_sa_quality_sweep.py",
    "03_scaling_analysis.py",
    "04_controls.py",
    "05_learning_dynamics.py",
]


def main():
    overall_start = time.time()
    results = []

    for stage in STAGES:
        path = os.path.join(HERE, stage)
        print("\n" + "=" * 78)
        print(f"  Running {stage}")
        print("=" * 78)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, path],
            cwd=HERE,
            check=False,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        results.append((stage, ok, elapsed))
        status = "OK" if ok else f"FAILED (rc={proc.returncode})"
        print(f"\n--> {stage}: {status} in {elapsed:.1f}s")

    total = time.time() - overall_start
    print("\n" + "=" * 78)
    print("  Pipeline summary")
    print("=" * 78)
    for stage, ok, elapsed in results:
        print(f"  {stage:35s}  {'OK' if ok else 'FAIL':5s}  {elapsed:8.1f}s")
    print(f"  total: {total:.1f}s")

    failed = [s for s, ok, _ in results if not ok]
    if failed:
        print(f"\n{len(failed)} stage(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
