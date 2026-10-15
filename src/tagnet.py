"""Steam tag white-space network: shared functions used by the analysis notebook.

Rules are fixed here before looking at results (see docs/PLAN.md section 3).
"""
import itertools
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEED_TAGS = {"Indie"}
# Release-model, adult-content and non-game tags: not a design choice for the studio's concept.
NON_DESIGN = {"Early Access", "Free to Play", "Sexual Content", "Nudity", "NSFW", "Hentai", "Mature",
              "Software", "Utilities", "Design & Illustration", "Game Development", "Audio Production",
              "Video Production", "Animation & Modeling", "Photo Editing", "Web Publishing",
              "Education", "Software Training", "Accounting"}

# Resolution: tags that name the same play mode are merged into one node (votes = the highest of the merged tags).
# 84% of Horror x Multiplayer games also carry a co-op tag, so keeping them apart would hide crowding.
RESOLVE = {t: "Multiplayer" for t in ["Co-op", "Online Co-Op", "Local Co-Op", "Co-op Campaign", "Local Multiplayer",
                                      "4 Player Local"]}

# min_tag, min_co, min_pair below 1 are shares of the games analysed (1.5%, 0.5%, 0.8% = 15, 5, 8 per 1,000 games)
DEFAULTS = dict(k=5, generic_df=0.25, min_tag=0.015, npmi_cut=0.10, min_co=0.005,
                min_pair=0.008, max_sat=1.5, max_overlap=0.2)


# ---------------------------------------------------------------- loading
def parse_tagstr(s):
    if not isinstance(s, str) or not s:
        return {}
    out = {}
    for part in s.split("|"):
        k, _, v = part.rpartition(":")
        out[k] = int(v) if v.isdigit() else 0
    return out


def _sorted(tags):
    if isinstance(tags, list):  # SteamSpy returns [] when a game has no tags
        tags = {}
    return dict(sorted(tags.items(), key=lambda kv: -kv[1]))


def load_games():
    """One row per sampled game: S1 frame fields + S2 (live Steam store) tags and review counts.

    SteamSpy fields (ss_*) are kept only to document that SteamSpy mirrors S1 and cannot triangulate.
    """
    frame = pd.read_csv(ROOT / "data/processed/frame.csv")
    rows = []
    for appid in frame["appID"]:
        p = ROOT / f"data/raw/steam/{appid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        row = {"appID": appid, "s2_tags": _sorted(d.get("tags") or {}),
               "s2_reviews": int(d.get("total_reviews") or 0), "fetched": d.get("_fetched_utc")}
        ss = ROOT / f"data/raw/steamspy/{appid}.json"
        if ss.exists():
            s = json.loads(ss.read_text())
            row["ss_tags"] = _sorted(s.get("tags") or {})
            row["ss_reviews"] = int(s.get("positive") or 0) + int(s.get("negative") or 0)
        rows.append(row)
    games = frame.merge(pd.DataFrame(rows), on="appID", how="inner")
    s1 = ROOT / "data/processed/s1_tags.csv"
    if s1.exists():
        t = pd.read_csv(s1)
        t["s1_tags"] = t["tags"].map(lambda s: dict(sorted(parse_tagstr(s).items(), key=lambda kv: -kv[1])))
        games = games.merge(t[["appID", "s1_tags"]], on="appID", how="left")
    return games


def load_cohort():
    """Every game in the cohort with its S1 (FronkonGames snapshot) tag votes and review total."""
    pop = pd.read_csv(ROOT / "data/processed/population.csv")
    t = pd.read_csv(ROOT / "data/processed/s1_tags.csv")
    t["tags"] = t["tags"].map(lambda s: dict(sorted(parse_tagstr(s).items(), key=lambda kv: -kv[1])))
    c = pop.merge(t, on="appID", how="inner")
    return c[c["tags"].map(len) > 0].reset_index(drop=True)


# ---------------------------------------------------------------- tag cleaning
def generic_tags(tag_lists, threshold):
    """Tags present in more than `threshold` share of games (seed and non-design removed first)."""
    n = len(tag_lists)
    df = Counter(t for tags in tag_lists for t in set(tags) if t not in SEED_TAGS | NON_DESIGN)
    return {t for t, c in df.items() if c / n > threshold}, {t: c / n for t, c in df.items()}


def top_k_tags(tag_dicts, k, generic):
    """Each game's top-k tags by vote, after dropping seed, non-design and generic tags."""
    drop = SEED_TAGS | NON_DESIGN | generic
    return [[t for t in tags if t not in drop][:k] for tags in tag_dicts]


# ---------------------------------------------------------------- graph
def cooccurrence(baskets):
    n_a = Counter(t for b in baskets for t in set(b))
    n_ab = Counter(tuple(sorted(p)) for b in baskets for p in itertools.combinations(set(b), 2))
    return n_a, n_ab


def npmi(n_ab, n_a, n_b, N):
    p_ab, p_a, p_b = n_ab / N, n_a / N, n_b / N
    if p_ab >= 1:
        return 1.0
    return np.log(p_ab / (p_a * p_b)) / -np.log(p_ab)


def build_graph(baskets, min_tag, npmi_cut, min_co, weight="npmi"):
    N = len(baskets)
    n_a, n_ab = cooccurrence(baskets)
    keep = {t for t, c in n_a.items() if c >= min_tag}
    G = nx.Graph()
    for (a, b), c in sorted(n_ab.items()):  # sorted: node/edge order must not depend on hash seed
        if a in keep and b in keep and c >= min_co:
            w = npmi(c, n_a[a], n_a[b], N)
            if weight == "npmi" and w < npmi_cut:
                continue
            G.add_edge(a, b, npmi=w, count=c, dist=1 - w if weight == "npmi" else 1 / c)
    for t in G:
        G.nodes[t]["n"] = n_a[t]
    return G, n_a, n_ab


def broker_table(G):
    bc = nx.betweenness_centrality(G, weight="dist", normalized=True)
    con = nx.constraint(G, weight="npmi")
    return (pd.DataFrame({"betweenness": bc, "constraint": con, "degree": dict(G.degree()),
                          "games": nx.get_node_attributes(G, "n")})
            .rename_axis("tag").reset_index()
            .sort_values(["betweenness", "tag"], ascending=[False, True]).set_index("tag"))


def overlap(G, a, b):
    """Neighbourhood overlap of a and b (excluding each other): 0 = they touch disjoint neighbourhoods."""
    na = set(G[a]) - {b} if a in G else set()
    nb = set(G[b]) - {a} if b in G else set()
    union = na | nb
    return len(na & nb) / len(union) if union else 0.0


# ---------------------------------------------------------------- white space
def pair_table(baskets, reviews, G, p):
    """Score every tag pair: supply (n_ab, saturation), performance (lift, hit rate), structure (overlap)."""
    N = len(baskets)
    reviews = np.asarray(reviews)
    med_all = np.median(reviews)
    hit_cut = np.quantile(reviews, 0.9)
    n_a, n_ab = cooccurrence(baskets)
    idx = {}
    for i, b in enumerate(baskets):
        for pr in itertools.combinations(sorted(set(b)), 2):
            idx.setdefault(pr, []).append(i)
    rows = []
    for (a, b), c in sorted(n_ab.items()):
        if n_a[a] < p["min_tag"] or n_a[b] < p["min_tag"] or c < p["min_pair"]:
            continue
        r = reviews[idx[(a, b)]]
        rows.append({"a": a, "b": b, "n_ab": c, "expected": n_a[a] * n_a[b] / N,
                     "saturation": c / (n_a[a] * n_a[b] / N),
                     "median_reviews": float(np.median(r)), "lift": float(np.median(r) / med_all),
                     "hit_rate": float((r >= hit_cut).mean()), "overlap": overlap(G, a, b),
                     "npmi": npmi(c, n_a[a], n_a[b], N)})
    t = pd.DataFrame(rows)
    t["eligible"] = (t["saturation"] <= p["max_sat"]) & (t["overlap"] <= p["max_overlap"])
    # deterministic order: ties in lift broken by support, then name
    return t.sort_values(["lift", "n_ab", "a", "b"], ascending=[False, False, True, True]).reset_index(drop=True)


def resolve(tags):
    """Merge same-play-mode tags (RESOLVE) into one, keeping the highest vote count; re-sort by votes."""
    out = {}
    for t, v in tags.items():
        c = RESOLVE.get(t, t)
        out[c] = max(out.get(c, 0), v)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def run(tag_dicts, reviews, resolve_tags=True, **kw):
    """Full pipeline for one parameter setting. Returns graph, brokers, pair table, generic set."""
    if resolve_tags:
        tag_dicts = [resolve(d) for d in tag_dicts]
    p = {**DEFAULTS, **kw}
    for key in ("min_tag", "min_co", "min_pair"):
        if p[key] < 1:
            p[key] = round(p[key] * len(tag_dicts))
    generic, _ = generic_tags([list(d) for d in tag_dicts], p["generic_df"])
    baskets = top_k_tags([list(d) for d in tag_dicts], p["k"], generic)
    G, _, _ = build_graph(baskets, p["min_tag"], p["npmi_cut"], p["min_co"])
    pairs = pair_table(baskets, reviews, G, p)
    return {"G": G, "pairs": pairs, "generic": generic, "baskets": baskets, "params": p}


def winners(pairs, top=3):
    e = pairs[pairs["eligible"]]
    return [tuple(sorted((r.a, r.b))) for r in e.head(top).itertuples()]
