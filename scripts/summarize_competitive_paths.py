"""Summarize the saved regularization paths for Section 4.2 and the response."""
import csv
import sys
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1]
directory = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/competitive_paths"
settings = [(model, m) for model in ("quantile", "logistic", "multinomial") for m in (128, 512)]


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


def report(title, cases, methods, repetitions):
    print(title)
    print("model\tm\tmethod\tmean seconds\tfits at tolerance")
    for model, dimension in cases:
        for method in methods:
            seconds, converged, fits = summarize(model, dimension, method, repetitions)
            print(f"{model}\t{dimension}\t{method}\t{seconds:.2f}\t{converged}/{fits}")
    print()


report("Section 4.2 and response, five matched instances", [("multinomial", 512)],
       ("LBFGS", "QMME", "Newton"), range(1, 6))
report("Section 4.2, all L-BFGS instances", settings, ("LBFGS",), range(1, 6))
report("Response Table R1, first matched instance", settings,
       ("QMME", "Newton", "LBFGS", "Newton-PCG"), (1,))
