"""Source S2: live SteamSpy pull for every sampled game (tags with votes, reviews, owners).

API docs: https://steamspy.com/api.php  - documented poll rate 1 request/second.
robots.txt: "User-agent: * / Allow: /". Game-level data only, no personal data.
Resumable: responses are cached as data/raw/steamspy/{appid}.json.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/steamspy"
URL = "https://steamspy.com/api.php"
HEADERS = {"User-Agent": "MADT7202-student-research/1.0 (academic coursework; 1 req/s)"}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(ROOT / "data/processed/frame.csv")["appID"].tolist()
    todo = [a for a in ids if not (CACHE / f"{a}.json").exists()]
    print(f"{len(ids)} games, {len(todo)} to fetch")
    for i, appid in enumerate(todo, 1):
        for attempt in range(3):
            try:
                r = requests.get(URL, params={"request": "appdetails", "appid": appid},
                                 headers=HEADERS, timeout=30)
                r.raise_for_status()
                data = r.json()
                data["_fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                (CACHE / f"{appid}.json").write_text(json.dumps(data))
                break
            except Exception as e:  # network hiccup: back off and retry
                print("retry", appid, e)
                time.sleep(5 * (attempt + 1))
        time.sleep(1.05)
        if i % 100 == 0:
            print(i, "done")
    print("finished")


if __name__ == "__main__":
    main()
