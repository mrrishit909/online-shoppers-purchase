"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_new_vs_returning.png, 02_months.png, 03_pagevalues.png, 04_models.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def visitors(ax):
    c = pd.read_csv(R / "conversion.csv").set_index("VisitorType").loc[["New_Visitor", "Returning_Visitor", "Other"]]
    x = np.arange(3)
    ax.bar(x, c["rate_pct"], color=[ORANGE, BLUE, GRAY], width=0.6, yerr=[c["rate_pct"] - c["lo"], c["hi"] - c["rate_pct"]], ecolor=INK, error_kw={"linewidth": 1, "capsize": 4})
    for xi, (_, r) in zip(x, c.iterrows()):
        ax.text(xi, 1.2, f"{r['rate_pct']:.1f}%", ha="center", color=BG, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{int(s):,} sessions" for n, s in zip(["New visitors", "Returning visitors", "Other"], c["sessions"])], fontsize=9); ax.set_ylim(0, 32); ax.set_ylabel("% of sessions ending in a purchase"); ax.grid(axis="x", visible=False)
    ax.set_title("New visitors buy more often")


def models(ax):
    m = pd.read_csv(R / "models.csv"); labels = [f"{r['model'].replace('gradient boosting', 'boosting').replace('logistic regression', 'logistic')}\n{r['pagevalues']} PageValues" for _, r in m.iterrows()]
    x = np.arange(len(m)); w = 0.38
    ax.bar(x - w / 2, m["cv_auc"], width=w, color=BLUE, label="5-fold cross-validation"); ax.bar(x + w / 2, m["time_split_auc"], width=w, color=ORANGE, label="trained Feb-Sep, tested Oct-Dec")
    for xi, (_, r) in zip(x, m.iterrows()):
        ax.text(xi - w / 2, r["cv_auc"] + 0.01, f"{r['cv_auc']:.2f}", ha="center", color=INK, fontsize=9); ax.text(xi + w / 2, r["time_split_auc"] + 0.01, f"{r['time_split_auc']:.2f}", ha="center", color=INK, fontsize=9)
    ax.axhline(0.5, color=DIM, linestyle="--", linewidth=1); ax.text(3.45, 0.51, "guessing", color=DIM, fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC (1 is perfect, 0.5 is guessing)"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper right"); ax.set_title("Predicting a purchase, with and without PageValues")


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1.25]})
    visitors(a); models(b)
    for ax in (a, b):
        ax.title.set_fontsize(15)
    fig.suptitle("Who buys online? 12,330 shopping sessions", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)

    c = pd.read_csv(R / "conversion.csv").set_index("VisitorType"); l = pd.read_csv(R / "logit_new_visitor.csv"); fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    visitors(a); y = np.arange(len(l))[::-1]
    for yi, (_, r) in zip(y, l.iterrows()):
        b.plot([r["lo"], r["hi"]], [yi, yi], color=BLUE, linewidth=3); b.scatter([r["or_new_visitor"]], [yi], color=BLUE, s=60, zorder=3); b.text(r["hi"] + 0.05, yi, f"{r['or_new_visitor']:.2f}", va="center", color=INK, fontsize=9)
    b.axvline(1, color=DIM, linestyle="--", linewidth=1); b.set_yticks(y); b.set_yticklabels([m.replace(" (a leaky feature)", "") for m in l["model"]], fontsize=8); b.set_xlim(0.8, 3); b.set_ylim(-0.6, 3.6)
    b.set_xlabel("odds of buying, new against returning visitors"); b.grid(axis="y", visible=False); b.set_title("...and the gap survives adjustment")
    save(fig, "01_new_vs_returning.png")

    m = pd.read_csv(R / "conversion_by_month.csv"); fig, ax = plt.subplots(figsize=(10, 3.8)); x = np.arange(len(m))
    ax.bar(x, m["rate_pct"], color=[ORANGE if v == m["rate_pct"].max() else BLUE for v in m["rate_pct"]], width=0.7, yerr=[m["rate_pct"] - m["lo"], m["hi"] - m["rate_pct"]], ecolor=INK, error_kw={"linewidth": 1, "capsize": 3})
    for xi, (_, r) in zip(x, m.iterrows()):
        ax.text(xi, 1, f"{int(r['sessions']):,}", ha="center", color=BG, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(m["Month"]); ax.set_ylabel("% of sessions ending in a purchase"); ax.set_ylim(0, 31); ax.grid(axis="x", visible=False)
    ax.set_title("Purchase rate by month (sessions at the foot of each bar; no January or April in the data)")
    save(fig, "02_months.png")

    a = pd.read_csv(R / "audit.csv").set_index("check")["value"]; fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.barh([1, 0], [a["purchase rate when PageValues is 0 (%)"], a["purchase rate when PageValues is above 0 (%)"]][:1] + [a["purchase rate when PageValues is above 0 (%)"]], color=[BLUE, ORANGE], height=0.6)
    for yi, v, n in [(1, a["purchase rate when PageValues is 0 (%)"], 9600), (0, a["purchase rate when PageValues is above 0 (%)"], int(a["sessions with PageValues above 0"]))]:
        ax.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK, fontsize=11)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["PageValues = 0", f"PageValues > 0 ({int(a['sessions with PageValues above 0']):,} sessions)"]); ax.set_xlim(0, 70); ax.set_xlabel("% of sessions ending in a purchase"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Sessions with a positive PageValues buy {a['purchase rate when PageValues is above 0 (%)'] / a['purchase rate when PageValues is 0 (%)']:.0f} times as often")
    save(fig, "03_pagevalues.png")

    fig, ax = plt.subplots(figsize=(10, 4.0)); models(ax); save(fig, "04_models.png")


if __name__ == "__main__":
    main()
