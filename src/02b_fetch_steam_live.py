"""Source S2 (live): Steam's own store data for every sampled game.

Why: SteamSpy (02_fetch_steamspy.py) turned out to mirror S1 exactly (same snapshot: top-5 tag Jaccard 1.0,
identical review counts), so it cannot triangulate. Steam's store is the primary upstream.

Per game, two requests:
  * https://store.steampowered.com/appreviews/{appid}?json=1  -> total positive/negative reviews (live)
  * https://store.steampowered.com/app/{appid}/               -> user tags with vote counts (InitAppTagModal)
robots.txt (store.steampowered.com) disallows only /share/, /news/externalpost/, account/login/join token
URLs, /email/ and /widget/ - /app/ and /appreviews/ are permitted. Game-level data only; no reviewer
identities or review text is stored. ~1.5 s between requests, back-off on HTTP 429.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/steam"
S = requests.Session()
S.headers["User-Agent"] = "MADT7202-student-research/1.0 (academic coursework; ~1 req/1.5s)"
S.cookies.update({"birthtime": "568022401", "lastagecheckage": "1-January-1988", "mature_content": "1",
                  "wants_mature_content": "1"})
TAG_RE = re.compile(r"InitAppTagModal\(\s*\d+,\s*(\[.*?\])\s*,", re.S)


def get(url, **params):
    for attempt in range(5):
        r = S.get(url, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"gave up on {url}")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(ROOT / "data/processed/frame.csv")["appID"].tolist()
    todo = [a for a in ids if not (CACHE / f"{a}.json").exists()]
    print(f"{len(ids)} games, {len(todo)} to fetch", flush=True)
    for i, appid in enumerate(todo, 1):
        try:
            q = get(f"https://store.steampowered.com/appreviews/{appid}", json=1, language="all",
                    purchase_type="all", num_per_page=0, filter_offtopic_activity=0).json()["query_summary"]
            time.sleep(1.5)
            page = get(f"https://store.steampowered.com/app/{appid}/", l="english").text
            m = TAG_RE.search(page)
            tags = {t["name"]: t["count"] for t in json.loads(m.group(1))} if m else {}
            (CACHE / f"{appid}.json").write_text(json.dumps({
                "appid": appid, "positive": q.get("total_positive"), "negative": q.get("total_negative"),
                "total_reviews": q.get("total_reviews"), "tags": tags,
                "_fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        except Exception as e:
            print("failed", appid, e, flush=True)
        time.sleep(1.5)
        if i % 100 == 0:
            print(i, "done", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
