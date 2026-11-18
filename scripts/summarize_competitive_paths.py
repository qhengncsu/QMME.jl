"""Summarize the saved regularization paths for Section 4.2 and the response."""
import csv
import sys
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1]
directory = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/competitive_paths"
models = ("quantile", "logistic", "multinomial")


def summarize(model, dimension, method, repetitions):
    seconds = []
    converged = 0
    fits = 0
    for rep in repetitions:
        path = directory / f"{model}_m{dimension}_rep{rep}_{method}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 30 or len({row["lambda"] for row in rows}) != 30:
            raise ValueError(f"Expected a complete 30-penalty path in {path}")
        # Both timing fields are path totals repeated on each penalty row.
        seconds.append(float(rows[0]["kernel_seconds"]) + float(rows[0]["solver_seconds"]))
        converged += sum(float(row["gradient"]) < 1e-4 for row in rows)
        fits += len(rows)
    return mean(seconds), converged, fits


def report(title, methods, repetitions):
    print(title)
    print("model\tm\tmethod\tmean seconds\tfits at tolerance")
    for model in models:
        for dimension in (128, 512):
            for method in methods:
                seconds, converged, fits = summarize(model, dimension, method, repetitions)
                print(f"{model}\t{dimension}\t{method}\t{seconds:.2f}\t{converged}/{fits}")
    print()


report("Section 4.2, matched instances", ("QMME", "Newton", "LBFGS"), range(1, 4))
report("Section 4.2, all L-BFGS instances", ("LBFGS",), range(1, 6))
report("Response, matched instance", ("QMME", "Newton", "LBFGS", "Newton-PCG"), (1,))
