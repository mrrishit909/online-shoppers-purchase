"""Step 1: the UCI "Online Shoppers Purchasing Intention" dataset (Sakar et al. 2019): 12,330 website sessions, one row each, with whether the session ended in a purchase.

    ./venv/bin/python download.py   -> data/online_shoppers_intention.csv, results/pull.json

UCI Machine Learning Repository, dataset 468, licence CC BY 4.0.
"""
import hashlib
import io
import json
import ssl
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import certifi

HERE = Path(__file__).parent
URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
CTX = ssl.create_default_context(cafile=certifi.where())


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    blob = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"}), context=CTX, timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
    raw = z.read(csv_name)
    (HERE / "data" / "online_shoppers_intention.csv").write_bytes(raw)
    info = {"pulled": date.today().isoformat(), "url": URL, "files_in_zip": z.namelist(), "rows": raw.count(b"\n") - 1, "sha256": hashlib.sha256(raw).hexdigest()}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
