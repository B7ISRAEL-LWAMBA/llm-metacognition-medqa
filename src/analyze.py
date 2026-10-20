"""Build the results table, model comparisons and the three figures.

Usage: python src/analyze.py
Writes results/results_table.{csv,md}, results/comparisons.csv and figures/*.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without opening windows

import pandas as pd

import plots
from metrics import auroc, accuracy, load_results, paired_bootstrap_diff, results_table

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"


def comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Paired differences that answer the project's questions: does C beat A,
    does B beat A, and does one model beat the other in each condition?"""
    ok = df[~df["parse_error"].astype(bool)]
    sets = {k: g for k, g in ok.groupby(["model_key", "condition"])}
    pairs = []
    for model in sorted(ok["model_key"].unique()):
        pairs += [((model, "C"), (model, "A")), ((model, "B"), (model, "A"))]
    models = sorted(ok["model_key"].unique())
    if len(models) == 2:
        pairs += [((models[0], c), (models[1], c)) for c in "ABC"]

    rows = []
    for a, b in pairs:
        if a not in sets or b not in sets:
            continue
        for name, metric in [("accuracy", accuracy), ("auroc", auroc)]:
            r = paired_bootstrap_diff(metric, sets[a], sets[b])
            rows.append({"comparison": f"{a[0]} {a[1]} vs {b[0]} {b[1]}", "metric": name, **r,
                         "ci_excludes_0": r["ci_low"] > 0 or r["ci_high"] < 0})
    return pd.DataFrame(rows)


def main():
    df = load_results()
    if df.empty:
        print("No results in results/raw yet. Run src/run.py first.")
        return
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    table = results_table(df)
    table.to_csv(RESULTS / "results_table.csv", index=False)
    (RESULTS / "results_table.md").write_text(table.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    comp = comparisons(df)
    if not comp.empty:
        comp.to_csv(RESULTS / "comparisons.csv", index=False)
        print("\n" + comp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    plots.reliability_diagram(df, FIGURES / "reliability_diagram.png")
    plots.confidence_by_correctness(df, FIGURES / "confidence_by_correctness.png")
    plots.risk_coverage_plot(df, FIGURES / "risk_coverage.png")
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
