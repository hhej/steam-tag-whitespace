"""Build the sampling frame from the FronkonGames Steam Games Dataset (MIT licence).

Source S1: https://huggingface.co/datasets/FronkonGames/steam-games-dataset
  data/train-00000-of-00001.parquet -> data/external/fronkon_games.parquet

Cohort: games whose Steam genre list contains "Indie", released 1 Jan 2023 - 31 Dec 2024,
with >= 10 user reviews (positive + negative). Every game has >= 21 months of reviews by Oct 2026.
Output: data/processed/population.csv (whole cohort) and data/processed/frame.csv (random sample).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external/fronkon_games.parquet"
OUT = ROOT / "data/processed"
SEED = 7202
SAMPLE_N = 1000
MIN_REVIEWS = 10

COLS = ["appID", "name", "release_date", "positive", "negative", "estimated_owners", "price",
        "genres", "developers"]


def main():
    df = pd.read_parquet(SRC, columns=COLS)
    print("rows in dataset:", len(df))
    df["appID"] = df["appID"].astype(int)
    df["release"] = pd.to_datetime(df["release_date"], format="mixed")
    df["reviews"] = df["positive"] + df["negative"]
    df["genres"] = df["genres"].map(lambda g: [] if g is None else list(g))
    df["developers"] = df["developers"].map(lambda g: "" if g is None else "|".join(g))
    is_indie = df["genres"].map(lambda g: "Indie" in g)
    in_window = df["release"].between("2023-01-01", "2024-12-31")
    pop = df[is_indie & in_window & (df["reviews"] >= MIN_REVIEWS)].drop_duplicates("appID").copy()
    print("cohort (population):", len(pop))

    sample = pop.sample(n=SAMPLE_N, random_state=SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["appID", "name", "release", "positive", "negative", "reviews", "estimated_owners", "price",
            "developers", "genres"]
    for frame, name in [(pop, "population.csv"), (sample, "frame.csv")]:
        f = frame[keep].copy()
        f["genres"] = f["genres"].map("|".join)
        f.to_csv(OUT / name, index=False)
    print("wrote", OUT / "population.csv", "and", OUT / "frame.csv")


if __name__ == "__main__":
    main()
