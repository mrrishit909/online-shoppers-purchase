# Who buys online?

12,330 website sessions of an online shop, each marked with whether it ended in a purchase. Three questions:
- do new visitors and returning visitors buy at different rates, and does the gap survive once you account for when they come and how they browse;
- what else separates buying sessions (month, weekend);
- how well can a model predict a purchase, and how much of that is real.

The questions follow DataCamp's "Will This Customer Purchase Your Product?" project. The data is the UCI Online Shoppers Purchasing Intention dataset, a real set of
sessions from one e-commerce site, not a course file.

Build log: https://mrrishit909.github.io/projects/online-shoppers-purchase/

## Data

UCI Machine Learning Repository, "Online Shoppers Purchasing Intention Dataset" (dataset 468; Sakar, Polat, Katircioglu and Kastro, *Neural Computing and Applications*,
2019), licence CC BY 4.0. One row per session, over one year: pages visited and time spent by page type, bounce and exit rates, "PageValues", month, weekend, visitor type
and the outcome (Revenue). 12,330 sessions; January and April are not in the data. 1.0 MB, so it is committed in `data/`; `download.py` refreshes it.

```
./venv/bin/pip install pandas numpy scipy statsmodels scikit-learn certifi      # in a venv
./venv/bin/python download.py    # data/online_shoppers_intention.csv
./venv/bin/python analyze.py     # results/*
./venv/bin/python check.py
./venv/bin/python charts.py
```

## Steps

1. **Download** the table.
2. **Audit:** duplicates, missing values, which months exist, and how the purchase rate behaves when `PageValues` is zero or positive.
3. **Who buys:** purchase rate with Wilson intervals by visitor type, month and weekend; a two-proportion z-test of new against returning visitors.
4. **Does the gap survive?** The odds ratio for a new visitor from four logistic regressions: crude, then adding month, weekend and special day, then browsing behaviour, then PageValues.
5. **Predict:** logistic regression and gradient boosting, each with and without PageValues, scored by 5-fold cross-validation and by a time split (train February to September,
   test October to December). Scores: AUC, average precision, and the share of buyers captured in the top tenth of predicted sessions.
6. **Check.** `check.py` recomputes all 136 numbers: the tables and tests from the raw CSV with textbook formulas, the four logistic fits with a hand-written Newton-Raphson, and every
   model score from the saved predictions with rank and sort formulas. The models are scikit-learn's, so they are not refitted.
7. **Charts.**

## Results

- **New visitors buy more often.** 24.9% of the 1,694 new-visitor sessions ended in a purchase (95% interval 22.9 to 27.0), against 13.9% of 10,551 returning ones (13.3 to 14.6): a
  gap of 11.0 points (8.8 to 13.1; z = 11.6). Overall 15.5% of sessions purchase.
- **The gap survives adjustment.** The odds ratio for a new visitor is 2.05 crude, 1.92 with month, weekend and special day, 1.58 with browsing behaviour, and 1.93 with PageValues:
  new visitors are still more likely to buy than returning ones with similar browsing.
- **Season and weekend.** The purchase rate is 10% to 11% in March, May and June, rises to 17.6% in August, 21.0% in October and 25.4% in November (2,998 sessions), then 12.5% in December; February has
  3 purchases in 184 sessions. Weekend sessions purchase 17.4% against 14.9%.
- **PageValues does the work.** Sessions with a positive PageValues (2,730 of them) buy 56.3% of the time, against 3.9% when it is zero. The dataset defines it as the average value of the
  pages visited before a transaction, so it is computed from purchases.
- **Prediction, with and without it.** With PageValues, gradient boosting reaches an AUC of 0.93 in cross-validation and its top tenth of sessions captures 50.3% of buyers. Without it the
  same model reaches 0.78 and captures 26.8%. On a time split (train February to September, test October to December) the scores are 0.84 with PageValues and 0.66 without, which is
  worse than logistic regression without it (0.68).
- **Data quirks:** 125 sessions are exact duplicates (the purchase rate is 15.6% without them, against 15.5% with); 38 have no product pages.

## Not done, and caveats

- One shop for one year, and the months present are not a full year. The time split mixes a change of months with a change of customers: the test months hold 5,274 of the 12,330 sessions, 2,998 of them in November.
- PageValues is treated as leaky because of how it is defined; the data cannot show how it was computed for each session. Dropping it is the conservative reading.
- "New against returning" is an association. Nothing here shows that being new makes a visitor more likely to buy; they may have arrived through different campaigns; the data has a traffic-type code but no key to it.
- The models use default settings with no tuning beyond a fixed seed; without PageValues, boosting is ahead of logistic regression in cross-validation (0.78 against 0.75) and behind it on the time split (0.66 against 0.68).
