# Plan — "Cozy Meets Cosmic Horror": Steam Tag White-Space Network (MADT 7202 Mid-Term)

## Context
- Assignment: one A4 PDF page (≥10pt, ≥1.5 cm margins), 8 required sections, 30-pt rubric, due **16 Oct 2026**. Individual; domain must differ from team's (Finance / SCB DataX).
- Goal: **30/30**. Sample C lost its 1 point for never justifying the co-mention *window* — we fix that explicitly.
- Decisions taken with user: hypothetical Bangkok indie studio as client · HTML→PDF via headless Chrome · public GitHub repo via `gh` · witty title, sober body.

## 1. Stress test — verdict: PASS (with a differentiation requirement)
| Test | Result |
|---|---|
| Is it already done? | Partially. Li & Zhang (2020) built a Steam tag co-occurrence network (306 nodes, Louvain, descriptive). Quantic Foundry (2018) visualised 2,000 games' tags. A Medium post does tag-level supply/attention trends. **None** ties pair-level combinations to commercial performance + structural position for a build decision. → Topic known, our *decision framing is novel*. Mention prior work in one clause to show awareness. |
| Data permitted? | SteamSpy API: documented, `robots.txt: Allow: /`, rate 1 req/s (verified live: returns tags w/ votes, positive, negative, owners). Steam store `robots.txt` does not disallow `/api/` or `/appreviews/` (verified live). FronkonGames Steam Games Dataset (Hugging Face/Kaggle) is **MIT-licensed**, ~124k games, has release_date, tags, positive/negative. Game-level data only → no personal data → PDPA not triggered. |
| Seed/generic trap present? | Yes, and it's a *feature*: "Indie" is the seed (100% of cohort); "Singleplayer", "Casual", "2D", "Adventure" are generic. We show the trap and remove it with a stated rule. |
| Can it hit every Advanced extra? | Framing: named owner + "if it came out the other way". Data: 2 sources triangulated + stated bias. Construction: NPMI vs raw count **and** top-5 vs top-20 tag window. Insight: cutoff sweep + bootstrap + limitation that flips the call. ✔ |
| Feasible in 2 days? | 1,000 SteamSpy calls ≈ 17 min. Rest is local compute. ✔ |

## 2. The page content (target text, numbers filled from analysis)
1. **Decision** — Which two-tag game concept the creative director of *a hypothetical 5-person Bangkok indie studio* pitches at the greenlight meeting for its next 18-month project. Changes the prototype, art direction, and Steam page tags.
2. **Why a network** — Counts say which tags are popular; they cannot say which *combination* bridges two audience neighbourhoods that rarely meet (a structural hole) yet perform well when they do. That's a path/neighbourhood question.
3. **Data table** — Sources (S1 frame: FronkonGames MIT dataset; S2: live SteamSpy pull, date), legal basis, period (games released 1 Jan 2023–31 Dec 2024, observed Oct 2026 ⇒ ≥21 months of reviews each), volume (population N, sample 1,000, seed 7202), triangulation number (S1 vs S2 top-5 tag Jaccard, review-count Spearman ρ), sampling bias (≥10 reviews filter over-represents games that got traction; reviews ≈ buyers, not unmet demand; English-centric tagging).
4. **Construction table** — Node = tag; edge = both tags in the same game's top-5 non-generic tags; weight = NPMI; pruning = NPMI ≥ 0.10 & co-occur ≥ 5 games, tag support ≥ 15 games; excluded: seed "Indie", generic tags (doc-freq > 25%, listed), non-design tags (Early Access, Free to Play, adult-content tags); final node/edge counts. **Two named alternatives with shown effect**: (a) raw count → "X of top-10 edges involve the 5 most common tags; NPMI: 0"; (b) top-20 window → recommendation unchanged/changed + why top-5 (store-capsule visibility, highest vote consensus).
5. **Figure** — Backbone of ~25–30 tags; node size = #games (supply), node colour = median reviews (demand), edge width = NPMI; recommended pair highlighted orange with a dashed "bridge" annotation; fixed-seed layout; labelled, not a hairball.
6. **Findings (3, each numeric)** — F1 structure (broker tag: betweenness/Burt constraint, which neighbourhoods it joins); F2 white space (pair A×B: n_AB games vs expected, median reviews ×cohort, hit rate vs cohort); F3 robustness (holds top-3 in X/100 bootstraps and at NPMI cutoffs 0.05/0.10/0.15/0.20, and K=5 vs 20; where it breaks).
7. **Actions (calendar-able)** — e.g. greenlight pitch on a fixed date; 6-week vertical slice; Steam "Coming Soon" page with A, B in top-5 tags + 30-day wishlist test with a numeric kill threshold → fall back to pair #2.
8. **What would change my mind** — Reviews measure buyers of games that exist, not unmet demand; small n_AB may be a few lucky hits. The wishlist test settles it; if lift collapses under a stricter cutoff or the top-20 window, switch to the "fast follower" pair (best performer among saturated combos). This is also the "if it came out the other way" statement.

## 3. Analysis rules (pre-registered before looking at results)
- **Generic rule**: drop tags with document frequency > 25% in the cohort (after removing seed). Then take each game's top-5 remaining tags by vote count.
- **NPMI**: `log(p_ab/(p_a p_b)) / −log p_ab`, p over games.
- **Centrality/structure** (Wk 7/8 only, two metrics): betweenness with distance = 1 − NPMI; Burt's constraint (low = broker). No community detection.
- **White-space score for pair (A,B)** with each tag ≥15 games:
  - supply: n_AB and saturation = n_AB / E[n_AB] (E = n_A·n_B/N)
  - performance: median total reviews of the n_AB games ÷ cohort median; hit rate = share ≥ cohort 90th-percentile reviews
  - eligible: 8 ≤ n_AB, saturation ≤ 1.5, and A,B ≥ 2 hops apart in the backbone without their direct edge (genuine bridge)
  - rank by performance lift; report bootstrap 90% CI.
- **Fallback rule**: if no pair is eligible or none survives robustness → recommend best-performing saturated pair ("fast follower"). Report says so either way.
- **Robustness**: NPMI cutoffs {0.05, 0.10, 0.15, 0.20} × min co-occur {5, 10}; window K ∈ {5, 10, 20}; 100 bootstrap resamples of games; S1-tags vs S2-tags re-run.

## 4. Implementation
Working dir `/Users/pollot/Documents/kg-midterm` (git init → public repo `steam-tag-whitespace` via `gh`).
```
pyproject.toml            # uv, Python 3.12: pandas pyarrow networkx matplotlib numpy scipy requests jupyter nbconvert pypdf
README.md                 # question, sources+licences, how to reproduce, link to PDF
docs/PLAN.md              # copy of this plan
src/01_frame.py           # download HF parquet (needed cols only) → filter cohort → sample 1,000 (seed 7202) → data/processed/frame.csv
src/02_fetch_steamspy.py  # 1 req/s, honest User-Agent, cache data/raw/steamspy/{appid}.json, resumable
notebooks/analysis.ipynb  # triangulation → tag cleaning → graph → metrics → white space → robustness → figure → tables
outputs/                  # fig_network.png/svg, tables/*.csv, numbers.json (every number on the page)
report/report.html → report/report.pdf   # headless Chrome --print-to-pdf, A4, 10pt, 1.5cm
```
- Large parquet in `data/external/` is gitignored; commit frame.csv + SteamSpy cache (small) so the page is reproducible.
- `numbers.json` is the single source for every number on the page (checklist: "every number appears in figure or table").

## 5. Timeline
- **Wed 14 Oct (tonight)**: env, 01_frame, start 02_fetch (background ~17 min), triangulation + EDA.
- **Thu 15 Oct**: graph, metrics, white space, robustness, figure; draft report.html; fit to one page.
- **Fri 16 Oct AM**: checklist pass, push repo, export PDF, submit.

## 6. Risks & fallbacks
- HF parquet field shapes differ (tags dict vs list; date strings) → normalise both; verify on 5 rows.
- SteamSpy outage/throttle → S1 tags alone for the graph, Steam `appreviews` endpoint for a 100-game review check (still 2 sources).
- Too few eligible pairs → loosen n_AB to ≥5 *as a stated sweep*, else fallback rule.
- Page overflow → shrink figure height first, then compress tables; never <10pt.

## 7. Verification
- `uv run python src/01_frame.py && uv run python src/02_fetch_steamspy.py` reproduce data; `uv run jupyter nbconvert --execute notebooks/analysis.ipynb` runs clean end-to-end.
- Assert in notebook: "Indie" absent from ranked tables; no generic-list tag in top results; node/edge counts match the table.
- PDF: pypdf page count == 1; visual check of the rendered page (Read the PDF); font ≥10pt and 1.5 cm margins in CSS.
- Final rubric self-score against all four Advanced criteria + the "Before you submit" checklist.

## Sources (stress test)
- Li & Zhang 2020, https://www.researchgate.net/publication/339081814
- Quantic Foundry 2018, https://quanticfoundry.com/2018/01/24/visualizing-steam-tags-related/
- SteamSpy API docs, https://steamspy.com/api.php
- FronkonGames dataset (MIT), https://huggingface.co/datasets/FronkonGames/steam-games-dataset
- How To Market A Game on tags, https://howtomarketagame.com/2020/11/12/steam-101-how-to-tag-your-game/

## Changes made during execution (2026-10-15)
- **S2 switched from SteamSpy to the live Steam store.** SteamSpy returned exactly the S1 snapshot (top-5 tag
  Jaccard 1.00, identical review counts for all 1,000 games), so it could not triangulate. S2 is now Steam's own
  app pages (user tags with votes) + `appreviews` endpoint (review totals). SteamSpy cache kept as evidence.
- **S1 tags come from `games.json`**: the parquet build of the FronkonGames dataset ships an empty `tags` column.
- **Cohort Indie filter uses Steam's genre field** (developer-declared), since parquet has no tags.
- **Bootstrap lift computed directly** from the pair's games in each resample (otherwise resamples where n_AB < 8
  were silently counted as failures).
- **Bridge test = neighbourhood overlap ≤ 0.2** (the "≥2 hops without the direct edge" wording was trivially true).

## Revision after first read-through (2026-10-15)
- **Figure redrawn:**
  - Shows 16 tags with grey nodes; the recommended pair is orange.
  - No colour scale.
  - Labels are placed only where they touch no other label, node or edge. The layout is rotated until that holds, and the notebook asserts it.
  - The call-out sits next to the dashed line.
  - adjustText was removed.
- **Page text rewritten in plain language.** The invented studio details, real calendar dates and the wishlist threshold were removed. The actions are now three steps with no dates.
- **Data and construction tables follow the assignment's row list:**
  - Data: source, how obtained, legal basis, period, volume.
  - Construction: node, edge, weight, pruning, size.
- **Puzzle's betweenness is shown in the figure label**, so the findings' numbers are on the page.

## Second revision: full cohort (2026-10-15)
- **The network is now built from all 9,094 cohort games** (S1), not the random 1,000.
  - The 1,000-game sample put only 9 games behind the recommended pair. On the full cohort, Puzzle × Story Rich has 104 games and a lift of 1.3× (not 5.9×), so the earlier pick was small-sample noise.
  - S2 (the live scrape of the 1,000) is now the accuracy check on S1: Jaccard 0.87, ρ 0.98.
- **Thresholds are now shares of the games analysed:** tag ≥1.5%, edge ≥0.5% shared games, pair ≥0.8%. These are the same 15 / 5 / 8 per 1,000 games as before, so their meaning doesn't change with N.
- **New result:** Horror × Multiplayer.
  - 75 games (55 expected), 3.1× median reviews, p < 0.001.
  - #1 at all 8 NPMI cutoffs; lift > 1 in 100/100 bootstraps.
  - #1 in only 36/100 bootstraps: in 42 it falls under the 73-game floor, and in 17 it crosses the 1.5× crowding limit.
- **Plan B is now the runner-up eligible pair** (Pixel Graphics × RPG). The crowded comparison is the best pair that fails only the crowding rule (Management × Simulation).
- **Figure:**
  - Shows each focus tag's 2 strongest neighbours plus the shortest paths linking them.
  - Node colour = median reviews.
  - Plan B is drawn in purple.
- **Page additions:**
  - A 3-row pair table, so the findings' numbers appear in a table.
  - Actions with relative weeks.
  - The broker (betweenness) finding was dropped because no action followed from it.
