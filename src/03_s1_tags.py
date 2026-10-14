"""Extract S1 (FronkonGames snapshot) tag votes for the sampled games, for triangulation against S2.

games.json -> data/external/fronkon_games.json (the parquet build ships the tags column empty).
Output: data/processed/s1_tags.csv  (appID, tags as "tag:votes|tag:votes|...")
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ids = set(pd.read_csv(ROOT / "data/processed/frame.csv")["appID"].astype(str))
    with open(ROOT / "data/external/fronkon_games.json") as f:
        games = json.load(f)
    rows = []
    for appid, g in games.items():
        if appid in ids:
            tags = g.get("tags") or {}
            if isinstance(tags, list):
                tags = {t: 0 for t in tags}
            rows.append({"appID": int(appid), "tags": "|".join(f"{k}:{v}" for k, v in tags.items())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/processed/s1_tags.csv", index=False)
    print(f"{len(out)} of {len(ids)} sampled games found; {(out.tags != '').sum()} with tags")


if __name__ == "__main__":
    main()
