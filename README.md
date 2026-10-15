# Steam Tag White-Space Network — MADT 7202 Mid-Term

**Question:** which two-tag game concept should the creative director of an indie studio build next?

**Approach:** build a tag co-occurrence network from 1,000 randomly sampled Indie games released 2023–2024, weight edges
by NPMI, then look for under-used tag pairs: pairs whose tags sit in separate network neighbourhoods, are combined no more often
than chance, yet out-perform the cohort on reviews. The one-page report is [`report/report.pdf`](report/report.pdf).

## Data and permission
| Source | How obtained | Legal basis |
|---|---|---|
| S1 — [FronkonGames Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) | download (parquet + JSON) | MIT licence |
| S2 — live Steam store: `store.steampowered.com/app/{id}` (user tags with votes) + `/appreviews/{id}?json=1` (review totals) | ~1 request/1.5 s, honest User-Agent, cached JSON in `data/raw/steam/` | `robots.txt` disallows only share/news/account/login/email/widget paths; `/app/` and `/appreviews/` are permitted |
| (tested, dropped) [SteamSpy API](https://steamspy.com/api.php) | 1 request/s, cached in `data/raw/steamspy/` | Mirrors S1 exactly (top-5 tag Jaccard 1.00, identical review counts), so it is not an independent source |

Game-level data only; no usernames, review text or personal data is collected (PDPA not engaged).

## Reproduce
```bash
uv sync
# S1: put the dataset files in data/external/
curl -L -o data/external/fronkon_games.parquet https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/data/train-00000-of-00001.parquet
curl -L -o data/external/fronkon_games.json https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/games.json
uv run python src/01_frame.py           # cohort + seeded random sample (seed 7202)
uv run python src/02b_fetch_steam_live.py  # S2 live Steam pull (~60 min, resumable; cache is committed)
uv run python src/02_fetch_steamspy.py     # optional: SteamSpy pull, only to show it mirrors S1
uv run python src/03_s1_tags.py         # S1 tag votes for triangulation
uv run python src/make_notebook.py && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
uv run python src/04_report.py          # one-page PDF via headless Chrome
```

## Layout
- `src/tagnet.py` — all analysis rules (seed/generic removal, NPMI graph, brokers, white-space scoring), fixed before results
- `notebooks/analysis.ipynb` — evidence notebook; writes every number on the page to `outputs/numbers.json`
- `outputs/` — figure, tables (brokers, pairs, robustness sweep)
- `docs/PLAN.md` — the pre-registered plan
