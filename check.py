"""Step 5: the project's check. Recompute everything with different code from analyze.py: conversion tables and tests from the raw CSV with textbook formulas (Wilson intervals,
pooled z-test), the four logistic regressions by a hand-written Newton-Raphson on a design matrix built by hand, and every model score (AUC, average precision, top-decile
capture) from the saved predictions with rank and sort formulas written out. The models themselves are scikit-learn's, so the check does not refit them.

    ./venv/bin/python check.py
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LOGS = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]
Z = 1.959963984540054
n_ok = 0


def same(label, got, want, tol=1e-6, abs_tol=None):
    """relative tolerance by default; abs_tol for values the results round to a fixed number of decimals (half a unit in the last place)"""
    global n_ok
    n_ok += 1
    limit = abs_tol if abs_tol is not None else tol * max(1.0, abs(float(want)))
    assert abs(float(got) - float(want)) <= limit, f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def wilson(x, n):
    p = x / n
    c, h = (p + Z * Z / (2 * n)) / (1 + Z * Z / n), Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return 100 * (c - h), 100 * (c + h)


def irls(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        info = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(info, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-12:
            break
    p = 1 / (1 + np.exp(-X @ b))
    return b, np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))


def auc(y, p):
    """chance a random buyer outscores a random non-buyer, ties half: average ranks"""
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks, i = [0.0] * len(p), 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    return (sum(r for r, v in zip(ranks, y) if v) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def average_precision(y, p):
    """sum over distinct score thresholds of (recall gained) x (precision there), best scores first"""
    order = sorted(range(len(p)), key=lambda i: -p[i])
    tp = fp = 0
    total, ap, prev_recall, i = sum(y), 0.0, 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            tp += y[order[k]]
            fp += 1 - y[order[k]]
        recall = tp / total
        ap += (recall - prev_recall) * tp / (tp + fp)
        prev_recall, i = recall, j + 1
    return ap


def main():
    rows = list(csv.DictReader(open(HERE / "data" / "online_shoppers_intention.csv")))
    n = len(rows)
    y = np.array([1.0 if r["Revenue"] == "TRUE" else 0.0 for r in rows])
    a = {r["check"]: float(r["value"]) for r in table("audit")}
    same("sessions", n, a["sessions"], 0); same("duplicates", n - len({tuple(r.values()) for r in rows}), a["exact duplicate rows"], 0); same("purchases", y.sum(), a["purchases"], 0)
    same("purchase rate", 100 * y.mean(), a["purchase rate (%)"], abs_tol=0.0051); same("PageValues above 0", sum(float(r["PageValues"]) > 0 for r in rows), a["sessions with PageValues above 0"], 0)
    same("months", len({r["Month"] for r in rows}), a["months present (of 12)"], 0); same("no product pages", sum(int(r["ProductRelated"]) == 0 for r in rows), a["sessions with no product pages"], 0)
    pv0 = [i for i, r in enumerate(rows) if float(r["PageValues"]) == 0]; pv1 = [i for i, r in enumerate(rows) if float(r["PageValues"]) > 0]
    same("rate when PageValues is 0", 100 * y[pv0].mean(), a["purchase rate when PageValues is 0 (%)"], abs_tol=0.0051); same("rate when PageValues above 0", 100 * y[pv1].mean(), a["purchase rate when PageValues is above 0 (%)"], abs_tol=0.0051)
    uniq, seen = [], set()
    for i, r in enumerate(rows):
        k = tuple(r.values())
        if k not in seen:
            seen.add(k); uniq.append(i)
    same("rate without duplicates", 100 * y[uniq].mean(), a["purchase rate without the duplicates (%)"], abs_tol=0.0051)

    def rates(name, key):
        for r in table(name):
            norm = lambda row: ("1" if row[key] == "TRUE" else "0") if key == "Weekend" else row[key]       # the results write the weekend flag as 1/0
            idx = [i for i, row in enumerate(rows) if norm(row) == r[key]]
            x, m = int(y[idx].sum()), len(idx)
            lo, hi = wilson(x, m)
            same(f"{name} {r[key]} sessions", m, r["sessions"], 0); same(f"{name} {r[key]} purchases", x, r["purchases"], 0); same(f"{name} {r[key]} rate", 100 * x / m, r["rate_pct"], abs_tol=0.0051)
            same(f"{name} {r[key]} lo", lo, r["lo"], abs_tol=0.0051); same(f"{name} {r[key]} hi", hi, r["hi"], abs_tol=0.0051)
    rates("conversion", "VisitorType"); rates("conversion_by_month", "Month"); rates("conversion_weekend", "Weekend")

    new, ret = [i for i, r in enumerate(rows) if r["VisitorType"] == "New_Visitor"], [i for i, r in enumerate(rows) if r["VisitorType"] == "Returning_Visitor"]
    x1, n1, x2, n2 = y[new].sum(), len(new), y[ret].sum(), len(ret)
    p1, p2, pp = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
    zt = json.load(open(R / "ztest.json")); se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2); z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    same("z new rate", 100 * p1, zt["new_rate"]); same("z returning rate", 100 * p2, zt["returning_rate"]); same("z difference", 100 * (p1 - p2), zt["diff_points"])
    same("z diff lo", 100 * (p1 - p2 - Z * se), zt["diff_lo"]); same("z diff hi", 100 * (p1 - p2 + Z * se), zt["diff_hi"]); same("z statistic", z, zt["z"]); same("z p-value", math.erfc(abs(z) / math.sqrt(2)), zt["p_two_sided"], 1e-4)

    def design(level):
        X = []
        for r in rows:
            v = [1.0, 1.0 * (r["VisitorType"] == "New_Visitor"), 1.0 * (r["VisitorType"] == "Other")]
            if level >= 1:
                v += [1.0 * (r["Month"] == m) for m in MONTHS if m != "Aug"] + [1.0 * (r["Weekend"] == "TRUE"), float(r["SpecialDay"])]
            if level >= 2:
                v += [math.log1p(float(r[c])) for c in LOGS] + [float(r["BounceRates"]), float(r["ExitRates"])]
            if level >= 3:
                v += [math.log1p(float(r["PageValues"]))]
            X.append(v)
        return np.array(X)
    for level, r in enumerate(table("logit_new_visitor")):
        X = design(level)
        b, cov = irls(X, y)
        se_ = math.sqrt(cov[1, 1])
        same(r["model"] + " parameters", X.shape[1], r["parameters"], 0); same(r["model"] + " OR", math.exp(b[1]), r["or_new_visitor"], 2e-4)
        same(r["model"] + " lo", math.exp(b[1] - Z * se_), r["lo"], 2e-4); same(r["model"] + " hi", math.exp(b[1] + Z * se_), r["hi"], 2e-4)
        same(r["model"] + " p", math.erfc(abs(b[1] / se_) / math.sqrt(2)), r["p"], 1e-3)

    oof = list(csv.DictReader(open(R / "oof_predictions.csv")))
    assert [int(r["y"]) for r in oof] == [int(v) for v in y], "saved outcomes differ from the raw file"
    yv = [int(v) for v in y]
    for r in table("models"):
        key = f"p_{r['model'].split()[0]}_{r['pagevalues']}_pagevalues"
        p = [float(o[key]) for o in oof]
        late = [i for i, o in enumerate(oof) if o[key + "_time_split"] != ""]
        pl, yl = [float(oof[i][key + "_time_split"]) for i in late], [yv[i] for i in late]
        top = sorted(range(n), key=lambda i: -p[i])[: n // 10]
        tag = f"{r['model']} {r['pagevalues']} PageValues"
        same(tag + " cv auc", auc(yv, p), r["cv_auc"], abs_tol=5.1e-5); same(tag + " cv average precision", average_precision(yv, p), r["cv_average_precision"], abs_tol=5.1e-5)
        same(tag + " top decile", 100 * sum(yv[i] for i in top) / sum(yv), r["top_decile_captures_pct"], abs_tol=0.0051); same(tag + " time-split auc", auc(yl, pl), r["time_split_auc"], abs_tol=5.1e-5)
        same(tag + " time-split average precision", average_precision(yl, pl), r["time_split_average_precision"], abs_tol=5.1e-5)
        same(tag + " time-split sessions", len(late), sum(1 for row in rows if row["Month"] in ("Oct", "Nov", "Dec")), 0)
    print(f"check.py: {n_ok} recomputed numbers agree with results/")


if __name__ == "__main__":
    main()
