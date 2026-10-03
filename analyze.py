"""Steps 2-4: audit the sessions, compare new and returning visitors, test whether the gap survives adjustment, and predict purchases with and without a leaky feature.

    ./venv/bin/python analyze.py
        -> results/audit.csv  conversion.csv  conversion_by_month.csv  conversion_weekend.csv  ztest.json  logit_new_visitor.csv  models.csv  oof_predictions.csv
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

HERE = Path(__file__).parent
R = HERE / "results"
NUM = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CAT = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LOGS = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]


def wilson(x, n):
    z, p = 1.959963984540054, x / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def rate_table(d, by):
    g = d.groupby(by)["Revenue"].agg(["size", "sum"]).reset_index().rename(columns={"size": "sessions", "sum": "purchases"})
    g["rate_pct"] = (100 * g["purchases"] / g["sessions"]).round(2)
    ci = [wilson(x, n) for x, n in zip(g["purchases"], g["sessions"])]
    g["lo"], g["hi"] = [round(c[0], 2) for c in ci], [round(c[1], 2) for c in ci]
    return g


def design(d, level):
    """Logistic design with fixed reference groups (returning visitor, August): level 1 adds month, weekend, special day; 2 adds browsing behaviour; 3 adds PageValues."""
    X = pd.DataFrame({"New_Visitor": (d["VisitorType"] == "New_Visitor") * 1.0, "Other_Visitor": (d["VisitorType"] == "Other") * 1.0})
    if level >= 1:
        for m in MONTHS:
            if m != "Aug":
                X["Month_" + m] = (d["Month"] == m) * 1.0
        X["Weekend"], X["SpecialDay"] = d["Weekend"] * 1.0, d["SpecialDay"]
    if level >= 2:
        for c in LOGS:
            X["log1p_" + c] = np.log1p(d[c])
        X["BounceRates"], X["ExitRates"] = d["BounceRates"], d["ExitRates"]
    if level >= 3:
        X["log1p_PageValues"] = np.log1p(d["PageValues"])
    return sm.add_constant(X)


def models(d, with_pv):
    num = [c for c in NUM if with_pv or c != "PageValues"]
    lr = make_pipeline(ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)]), LogisticRegression(max_iter=3000))
    cat_idx = [len(num) + i for i in range(len(CAT))]
    hgb = make_pipeline(ColumnTransformer([("n", "passthrough", num), ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)]),
                        HistGradientBoostingClassifier(random_state=0, categorical_features=cat_idx))
    return {"logistic regression": lr, "gradient boosting": hgb}


def main():
    R.mkdir(exist_ok=True)
    d = pd.read_csv(HERE / "data" / "online_shoppers_intention.csv")
    d["Revenue"] = d["Revenue"].astype(int)
    for c in ("Weekend",):
        d[c] = d[c].astype(int)

    # Step 2: audit
    pd.DataFrame([("sessions", len(d)), ("exact duplicate rows", int(d.duplicated().sum())), ("missing values", int(d.isna().sum().sum())), ("purchases", int(d["Revenue"].sum())),
                  ("purchase rate (%)", round(100 * d["Revenue"].mean(), 2)), ("sessions with PageValues above 0", int((d["PageValues"] > 0).sum())), ("months present (of 12)", d["Month"].nunique()),
                  ("sessions with no product pages", int((d["ProductRelated"] == 0).sum())), ("purchase rate when PageValues is 0 (%)", round(100 * d.loc[d["PageValues"] == 0, "Revenue"].mean(), 2)),
                  ("purchase rate when PageValues is above 0 (%)", round(100 * d.loc[d["PageValues"] > 0, "Revenue"].mean(), 2)),
                  ("purchase rate without the duplicates (%)", round(100 * d.drop_duplicates()["Revenue"].mean(), 2))], columns=["check", "value"]).to_csv(R / "audit.csv", index=False)

    # Step 3: who buys: new against returning, month, weekend
    rate_table(d, "VisitorType").to_csv(R / "conversion.csv", index=False)
    mo = rate_table(d, "Month"); mo["order"] = mo["Month"].map({m: i for i, m in enumerate(MONTHS)}); mo.sort_values("order").drop(columns="order").to_csv(R / "conversion_by_month.csv", index=False)
    rate_table(d, "Weekend").to_csv(R / "conversion_weekend.csv", index=False)
    new, ret = d[d["VisitorType"] == "New_Visitor"], d[d["VisitorType"] == "Returning_Visitor"]
    x1, n1, x2, n2 = new["Revenue"].sum(), len(new), ret["Revenue"].sum(), len(ret)
    p1, p2, pp = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)); se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    json.dump({"new_rate": 100 * p1, "returning_rate": 100 * p2, "diff_points": 100 * (p1 - p2), "diff_lo": 100 * (p1 - p2 - 1.959963984540054 * se), "diff_hi": 100 * (p1 - p2 + 1.959963984540054 * se),
               "z": z, "p_two_sided": 2 * stats.norm.sf(abs(z))}, open(R / "ztest.json", "w"), indent=2)

    # does the gap survive? odds ratio for a new visitor, crude and adjusted step by step
    rows = []
    for level, name in [(0, "crude"), (1, "+ month, weekend, special day"), (2, "+ browsing behaviour"), (3, "+ PageValues (a leaky feature)")]:
        r = sm.Logit(d["Revenue"], design(d, level)).fit(disp=0, method="newton", maxiter=100)
        lo, hi = r.conf_int().loc["New_Visitor"]
        rows.append({"model": name, "or_new_visitor": float(np.exp(r.params["New_Visitor"])), "lo": float(np.exp(lo)), "hi": float(np.exp(hi)), "p": float(r.pvalues["New_Visitor"]), "parameters": len(r.params)})
    pd.DataFrame(rows).round(4).to_csv(R / "logit_new_visitor.csv", index=False)

    # Step 4: prediction, 5-fold cross-validation and a time split (train February-September, test October-December)
    y = d["Revenue"].to_numpy()
    folds = np.zeros(len(d), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(d, y)):
        folds[te] = k
    oof = pd.DataFrame({"row": np.arange(len(d)), "fold": folds, "month": d["Month"], "y": y})
    late = d["Month"].isin(["Oct", "Nov", "Dec"]).to_numpy()
    out = []
    for with_pv in (True, False):
        for name in ("logistic regression", "gradient boosting"):
            col = f"p_{name.split()[0]}_{'with' if with_pv else 'without'}_pagevalues"
            p = np.zeros(len(d))
            for k in range(5):
                m = models(d, with_pv)[name].fit(d[folds != k], y[folds != k]); p[folds == k] = m.predict_proba(d[folds == k])[:, 1]
            oof[col] = p
            m = models(d, with_pv)[name].fit(d[~late], y[~late]); pt = m.predict_proba(d[late])[:, 1]
            oof[col + "_time_split"] = np.nan; oof.loc[late, col + "_time_split"] = pt
            top = np.argsort(-p)[: len(d) // 10]
            out.append({"model": name, "pagevalues": "with" if with_pv else "without", "cv_auc": roc_auc_score(y, p), "cv_average_precision": average_precision_score(y, p),
                        "top_decile_captures_pct": 100 * y[top].sum() / y.sum(), "time_split_auc": roc_auc_score(y[late], pt), "time_split_average_precision": average_precision_score(y[late], pt)})
    oof.to_csv(R / "oof_predictions.csv", index=False)
    pd.DataFrame(out).round(4).to_csv(R / "models.csv", index=False)
    print(pd.read_csv(R / "audit.csv").to_string(index=False)); print(pd.read_csv(R / "conversion.csv").to_string(index=False)); print(open(R / "ztest.json").read())
    print(pd.read_csv(R / "conversion_by_month.csv").to_string(index=False)); print(pd.read_csv(R / "conversion_weekend.csv").to_string(index=False))
    print(pd.read_csv(R / "logit_new_visitor.csv").to_string(index=False)); print(pd.read_csv(R / "models.csv").to_string(index=False))


if __name__ == "__main__":
    main()
