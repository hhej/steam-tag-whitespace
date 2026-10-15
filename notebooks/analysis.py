# %% [markdown]
# # Steam tag network: which two-tag concept should an indie studio build next?
# MADT 7202 mid-term · evidence notebook. Every number on the one-page report is written to
# `outputs/numbers.json` by this notebook. Rules live in `src/tagnet.py`.
#
# **Decision:** which two-tag game concept the creative director of an indie studio builds next.
#
# **Sources:** S1 = FronkonGames Steam Games Dataset (MIT): every game in the cohort, with tag votes and review
# totals. This is the data the network is built from. S2 = live Steam store pull for a random 1,000 of those games
# (tags from the app page, review totals from `appreviews`), used to check that S1 is accurate.
# SteamSpy was also pulled but mirrors S1 exactly (shown below), so it is not an independent source.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import tagnet as tn

OUT = ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
NUM = {}  # every number used on the page
pd.set_option("display.width", 160)

cohort = tn.load_cohort()                      # S1: all games, the analysis set
games = tn.load_games()                        # S2: live pull for the random 1,000, the cross-check set
games = games[games["s2_tags"].map(len) > 0].reset_index(drop=True)
NUM.update(population=len(cohort), sampled=1000, analysed=len(games),
           fetched_from=games["fetched"].min()[:10], fetched_to=games["fetched"].max()[:10],
           median_reviews=float(cohort["reviews"].median()))
print(NUM)

# %% [markdown]
# ## 1 · Is S1 accurate? Compare it with the live store (S2) on the random 1,000

# %%
def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan

def top5(d):
    return [t for t in d if t not in tn.SEED_TAGS][:5]

j5 = [jacc(top5(a), top5(b)) for a, b in zip(games["s1_tags"], games["s2_tags"])]
rho = spearmanr(games["reviews"], games["s2_reviews"]).statistic
ss = games["ss_tags"].notna()
j5_ss = [jacc(top5(a), top5(b)) for a, b in zip(games.loc[ss, "s1_tags"], games.loc[ss, "ss_tags"])]
NUM.update(tri_top5_jaccard=float(np.nanmean(j5)), tri_top5_identical=float(np.mean(np.array(j5) == 1)),
           tri_reviews_spearman=float(rho),
           review_growth=float(games["s2_reviews"].sum() / games["reviews"].sum()),
           steamspy_vs_s1_jaccard=float(np.nanmean(j5_ss)),
           steamspy_vs_s1_reviews_equal=float((games.loc[ss, "ss_reviews"] == games.loc[ss, "reviews"]).mean()))
print({k: round(NUM[k], 3) for k in ["tri_top5_jaccard", "tri_top5_identical", "tri_reviews_spearman",
                                     "review_growth", "steamspy_vs_s1_jaccard", "steamspy_vs_s1_reviews_equal"]})

# %% [markdown]
# ## 2 · The seed and generic-tag trap
# "Indie" is the collection seed. Tags in > 25 % of games are generic (rule fixed in advance).

# %%
dicts = list(cohort["tags"])
reviews = cohort["reviews"].values
tag_lists = [list(d) for d in dicts]
indie_share = np.mean(["Indie" in t for t in tag_lists])
generic, dfreq = tn.generic_tags(tag_lists, tn.DEFAULTS["generic_df"])
gen_table = pd.Series(dfreq).sort_values(ascending=False).head(20).rename("doc_freq").to_frame()
gen_table["generic"] = gen_table.index.isin(generic)
NUM.update(indie_share=float(indie_share), generic=sorted(generic, key=lambda t: -dfreq[t]),
           n_generic=len(generic))
gen_table.round(3)

# %% [markdown]
# ## 3 · Graph construction
# K = 5 tags per game; tag on ≥ 1.5 % of games; edge NPMI ≥ 0.10 and ≥ 0.5 % shared games.
# (Thresholds are shares of the games analysed, so they mean the same thing at 1,000 or 9,094 games.)

# %%
R = tn.run(dicts, reviews)
G, pairs, P = R["G"], R["pairs"], R["params"]
n_tags_raw = len({t for d in dicts for t in d})
NUM.update(nodes=G.number_of_nodes(), edges=G.number_of_edges(), tags_before=n_tags_raw,
           min_tag=P["min_tag"], min_co=P["min_co"], min_pair=P["min_pair"])
print(f"{n_tags_raw} distinct tags before pruning -> {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", P)

# %% [markdown]
# ### Named alternative (a): raw co-occurrence count instead of NPMI
# Raw count is computed on the *uncleaned* top-5 tags (what a naive build sees).

# %%
naive = [[t for t in tags if t not in tn.SEED_TAGS][:5] for tags in tag_lists]
n_a, n_ab = tn.cooccurrence(naive)
top5_common = [t for t, _ in n_a.most_common(5)]
raw_top = sorted(n_ab.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
raw_touch = sum(a in top5_common or b in top5_common for (a, b), _ in raw_top)
npmi_top = sorted(G.edges(data=True), key=lambda e: (-e[2]["npmi"], e[0], e[1]))[:10]
npmi_touch = sum(a in top5_common or b in top5_common for a, b, _ in npmi_top)
NUM.update(top5_common=top5_common, raw_top10_touch_common=raw_touch, npmi_top10_touch_common=npmi_touch,
           raw_top_edge=list(raw_top[0][0]), raw_top_edge_count=raw_top[0][1])
print("5 most common tags:", top5_common)
print("raw-count top-10 edges:", raw_top)
print("NPMI top-10 edges:", [(a, b, round(d["npmi"], 2)) for a, b, d in npmi_top])

# %% [markdown]
# ## 4 · Pairs: which combinations perform, and which are already crowded?
# A pair is eligible if it is not over-supplied (≤ 1.5× the games chance predicts) and its two tags sit in
# separate neighbourhoods (neighbour overlap ≤ 0.2). Eligible pairs are ranked by lift (median reviews ÷ cohort).

# %%
pairs.to_csv(OUT / "tables/pairs.csv", index=False)
elig = pairs[pairs["eligible"]].reset_index(drop=True)
print(len(pairs), "scored pairs;", len(elig), "eligible")
elig.head(10).round(2)

# %%
# Crowded comparison: best pair that passes the overlap rule but fails only the crowding rule
crowded = pairs[(pairs["saturation"] > P["max_sat"]) & (pairs["overlap"] <= P["max_overlap"])].reset_index(drop=True)
top10 = pairs.head(10)
NUM.update(top10_lift_crowded=int((top10["saturation"] > P["max_sat"]).sum()))
print("of the 10 highest-lift pairs,", NUM["top10_lift_crowded"], "are over-supplied")
crowded.head(5).round(2)

# %% [markdown]
# ## 5 · Robustness

# %%
def rank_of(pair, pairs_df):
    e = pairs_df[pairs_df["eligible"]].reset_index(drop=True)
    for i, r in enumerate(e.itertuples(), 1):
        if tuple(sorted((r.a, r.b))) == pair:
            return i
    return None

best = tuple(sorted((elig.loc[0, "a"], elig.loc[0, "b"])))
runner = tuple(sorted((elig.loc[1, "a"], elig.loc[1, "b"])))
sweep = []
for cut in [0.05, 0.10, 0.15, 0.20]:
    for mc in [0.005, 0.01]:
        sweep.append({"param": f"NPMI>={cut}, co>={mc:.1%}",
                      "rank": rank_of(best, tn.run(dicts, reviews, npmi_cut=cut, min_co=mc)["pairs"])})
for k in [3, 5, 10, 20]:
    sweep.append({"param": f"K={k}", "rank": rank_of(best, tn.run(dicts, reviews, k=k)["pairs"])})
for gdf in [0.20, 0.25, 0.35]:
    sweep.append({"param": f"generic>{gdf:.0%}", "rank": rank_of(best, tn.run(dicts, reviews, generic_df=gdf)["pairs"])})
for mp in [0.006, 0.008, 0.010]:
    sweep.append({"param": f"pair>={mp:.1%}", "rank": rank_of(best, tn.run(dicts, reviews, min_pair=mp)["pairs"])})
sweep = pd.DataFrame(sweep)
p20 = tn.run(dicts, reviews, k=20)["pairs"]
r20 = p20[(p20.a.isin(best)) & (p20.b.isin(best))].iloc[0]
NUM.update(k20_n=int(r20.n_ab), k20_lift=float(r20.lift), k20_sat=float(r20.saturation))
sweep.to_csv(OUT / "tables/robustness_sweep.csv", index=False)
sweep

# %%
# Second source: the pair's lift on the live-scraped sample (S2 tags and live review counts)
s2_base = tn.run(list(games["s2_tags"]), games["s2_reviews"].values)["baskets"]
s2_rev = games["s2_reviews"].values
s2_has = np.array([set(best) <= set(b) for b in s2_base])
NUM.update(s2_pair_n=int(s2_has.sum()), s2_pair_lift=float(np.median(s2_rev[s2_has]) / np.median(s2_rev)))
print("S2 sample:", NUM["s2_pair_n"], "games with the pair, lift", round(NUM["s2_pair_lift"], 2))

# %%
# Bootstrap: resample games. Rank = is the pair still the top eligible pair? Lift = the pair's games' median
# reviews / cohort median. Saturation = does the pair cross the 1.5x crowding limit in the resample?
rng = np.random.default_rng(7202)
has_pair = np.array([set(best) <= set(b) for b in R["baskets"]])
boot_rank, boot_lift, boot_sat, boot_planb = [], [], [], []
for _ in range(100):
    i = rng.integers(0, len(cohort), len(cohort))
    pr = tn.run([dicts[j] for j in i], reviews[i])["pairs"]
    boot_rank.append(rank_of(best, pr))
    boot_planb.append(rank_of(runner, pr))
    row = pr[(pr.a == best[0]) & (pr.b == best[1])]
    boot_sat.append(float(row["saturation"].iloc[0]) if len(row) else np.nan)
    r_i, h_i = reviews[i], has_pair[i]
    boot_lift.append(np.median(r_i[h_i]) / np.median(r_i) if h_i.any() else np.nan)
boot_rank, boot_planb = pd.Series(boot_rank, dtype="float"), pd.Series(boot_planb, dtype="float")
boot_lift, boot_sat = np.array(boot_lift), np.array(boot_sat)
# why the pair is not #1 in a resample: below the support floor (not scored), over the crowding limit, or outranked
NUM.update(boot_below_floor=int(np.isnan(boot_sat).sum()),
           boot_outranked=int(((boot_rank > 1) | (boot_rank.isna() & ~np.isnan(boot_sat) & (boot_sat <= P["max_sat"]))).sum()),
           boot_planb_top1=int((boot_planb == 1).sum()), boot_either_top1=int(((boot_rank == 1) | (boot_planb == 1)).sum()))
NUM.update(boot_top1=int((boot_rank == 1).sum()), boot_top3=int((boot_rank <= 3).sum()),
           boot_lift_gt1=int((boot_lift > 1).sum()), boot_sat_over=int((boot_sat > P["max_sat"]).sum()),
           boot_lift_p05=float(np.nanpercentile(boot_lift, 5)), boot_lift_p95=float(np.nanpercentile(boot_lift, 95)))
print({k: NUM[k] for k in ["boot_top1", "boot_top3", "boot_lift_gt1", "boot_sat_over", "boot_below_floor",
                            "boot_outranked", "boot_planb_top1", "boot_either_top1", "boot_lift_p05", "boot_lift_p95"]})

# %%
# Permutation test: how often do n random games from the cohort reach the pair's median reviews?
b0 = elig.loc[0]
draws = np.array([np.median(rng.choice(reviews, int(b0.n_ab), replace=False)) for _ in range(10000)])
NUM.update(perm_p=float((draws >= b0.median_reviews).mean()), perm_draws=len(draws))
print("permutation p =", NUM["perm_p"])

# %% [markdown]
# ## 6 · Record the headline numbers

# %%
def rec(r):
    return dict(pair=[r.a, r.b], n=int(r.n_ab), expected=float(r.expected), sat=float(r.saturation),
                overlap=float(r.overlap), median=float(r.median_reviews), lift=float(r.lift), hit=float(r.hit_rate),
                edge=bool(G.has_edge(r.a, r.b)))

b = elig.loc[0]
NUM.update(best=list(best), runner=list(runner), pick=rec(b), planb=rec(elig.loc[1]), crowd=rec(crowded.loc[0]),
           hit_cut=float(np.quantile(reviews, 0.9)),
           sweep=sweep.to_dict("records"), eligible=len(elig), scored_pairs=len(pairs))
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=2, default=str))
pd.DataFrame([NUM["pick"], NUM["planb"], NUM["crowd"]]).round(2)

# %% [markdown]
# ## 7 · The figure: the neighbourhoods of the recommended pair and plan B
# Node colour = median reviews of games with the tag. Labels are placed only where they touch no other label,
# node or edge; the cell fails if that is impossible.

# %%
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

INK, EDGE, ACCENT, PLANB = "#0b0b0b", "#d3d2cc", "#eb6834", "#7a4fc9"
BLUES = LinearSegmentedColormap.from_list("seq", ["#cfe3f7", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

tag_med = {t: float(np.median(reviews[[t in bk for bk in R["baskets"]]])) for t in G}
focus = set(best) | set(runner)
# each focus tag's 2 strongest neighbours (by NPMI), plus the shortest paths that link the four focus tags
show = set(focus)
for t in sorted(focus):
    show |= set(sorted(G[t], key=lambda u: (-G[t][u]["npmi"], u))[:2])
for u, v in [best, runner, (best[0], runner[0])]:
    show |= set(nx.shortest_path(G, u, v, weight="dist"))
H = G.subgraph(show).copy()
assert nx.is_connected(H)
# rebuild in sorted order: subgraph views iterate a Python set, which would make the layout hash-seed dependent
H2 = nx.Graph()
H2.add_nodes_from((t, G.nodes[t]) for t in sorted(H))
H2.add_edges_from(sorted((*sorted((u, v)), d) for u, v, d in H.edges(data=True)))
H = H2
# candidate layouts: Kamada-Kawai spaced by hops (strong links would otherwise pull tags on top of each other),
# first from its default start, then from fixed-seed spring layouts. Edge width still shows NPMI.
layouts = [nx.kamada_kawai_layout(H, weight=None)] + [
    nx.kamada_kawai_layout(H, weight=None, pos=nx.spring_layout(H, seed=sd)) for sd in range(1, 41)]
nmax = max(H.nodes[t]["n"] for t in H)
sizes = {t: 40 + 360 * H.nodes[t]["n"] / nmax for t in H}
norm = LogNorm(vmin=min(tag_med[t] for t in H), vmax=max(tag_med[t] for t in H))
ring = {t: ACCENT for t in best} | {t: PLANB for t in runner}
pk = NUM["pick"]
msg = (f"{best[0]} × {best[1]}{'' if pk['edge'] else ': no edge'}\n"
       f"{pk['n']} games (exp. {pk['expected']:.0f}) · {pk['lift']:.1f}× median reviews")


def hit(a, c, pad=2):
    return not (a[2] + pad < c[0] or c[2] + pad < a[0] or a[3] + pad < c[1] or c[3] + pad < a[1])


def inside_pts(pts, box):
    return ((pts[:, 0] > box[0] + 1) & (pts[:, 0] < box[2] - 1) & (pts[:, 1] > box[1] + 1) & (pts[:, 1] < box[3] - 1)).any()


def draw(kk, angle, flip):
    """Draw the graph with the layout rotated/flipped; place labels where they touch no label, node or edge.
    Returns the figure and the number of labels that could only be placed on top of an edge."""
    th = np.radians(angle)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = np.array([kk[t] for t in H]) @ rot.T * [flip, 1]
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))  # stretch to the figure's aspect ratio
    pos = {t: (x * 2.4, y) for t, (x, y) in zip(H, xy)}

    fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=300)
    ax.set_xlim(-0.3, 2.7); ax.set_ylim(-0.1, 1.25); ax.axis("off")  # top strip kept free for the colour key
    nx.draw_networkx_edges(H, pos, ax=ax, width=[0.3 + 6.0 * d["npmi"] ** 1.5 for _, _, d in H.edges(data=True)],
                           edge_color=EDGE)
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=list(H), node_size=[sizes[t] for t in H],
                           node_color=[tag_med[t] for t in H], cmap=BLUES, vmin=norm.vmin, vmax=norm.vmax,
                           linewidths=[2.4 if t in ring else 0.8 for t in H],
                           edgecolors=[ring.get(t, "white") for t in H])
    pairs_drawn = [(best, ACCENT), (runner, PLANB)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    px = ax.transData.transform
    nodes = {}
    for t in H:
        r = np.sqrt(sizes[t]) / 2 * fig.dpi / 72
        cx, cy = px(pos[t])
        nodes[t] = (cx - r, cy - r, cx + r, cy + r)
    def clear_of_nodes(pts, ends):
        """True if no point of the line comes within a node (plus 15 px at 300 dpi) other than its two ends."""
        return all(np.hypot(*(pts - px(pos[w])).T).min() >= (nodes[w][2] - nodes[w][0]) / 2 + 15
                   for w in H if w not in ends)

    # real edges are straight and may not pass under another node (it would look as if that node sat on the edge)
    edge_pts = [px(np.linspace(pos[u], pos[v], 40)) for u, v in H.edges()]
    if not all(clear_of_nodes(p, e) for p, e in zip(edge_pts, H.edges())):
        return fig, 99, 9.0
    # the recommended pair and plan B have no edge: draw them dashed, bending (arc3) around any node in the way
    pair_mid = {}
    for (u, v), col in pairs_drawn:
        a, b2 = px(pos[u]), px(pos[v])
        for rad in (0, 0.2, -0.2, 0.35, -0.35, 0.5, -0.5):
            ctrl = (a + b2) / 2 + rad * np.array([b2[1] - a[1], a[0] - b2[0]])  # matplotlib's arc3 control point
            s_ = np.linspace(0, 1, 60)[:, None]
            pts = (1 - s_) ** 2 * a + 2 * (1 - s_) * s_ * ctrl + s_ ** 2 * b2
            if clear_of_nodes(pts, (u, v)):
                break
        else:
            return fig, 99, 9.0
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                     linestyle=(0, (4, 3)), color=col, lw=1.8, zorder=0.5, shrinkA=0, shrinkB=0))
        edge_pts.append(pts)
        pair_mid[(u, v)] = ax.transData.inverted().transform(pts[30])
    edge_px = np.vstack(edge_pts)

    # node colour scale: a small horizontal bar in whichever corner is clear of nodes and edges
    for corner in [(0.80, 0.95), (0.01, 0.95), (0.41, 0.95)]:
        cax = ax.inset_axes([corner[0], corner[1], 0.18, 0.025])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUES), cax=cax, orientation="horizontal")
        ticks = [t for t in [20, 30, 50, 100, 200, 500] if norm.vmin <= t <= norm.vmax]
        cb.set_ticks(ticks, labels=[str(t) for t in ticks]); cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        cb.outline.set_visible(False)
        cax.set_title("median reviews of games with the tag", fontsize=6.5, pad=2)
        cbx = cax.get_tightbbox(rend)
        cbox = (cbx.x0, cbx.y0, cbx.x1, cbx.y1)
        if (cbox[0] >= frame.x0 and cbox[2] <= frame.x1 and not inside_pts(edge_px, cbox)
                and not any(hit(cbox, nb) for nb in nodes.values())):
            break
        cax.remove()
    else:
        return fig, 99, 9.0
    nodes["_colourbar"] = cbox
    placed = {}

    def box_dist(box, w):
        cx, cy = px(pos[w])
        return np.hypot(max(box[0] - cx, 0, cx - box[2]), max(box[1] - cy, 0, cy - box[3]))

    def free(box, own=None, edges=True):
        # a label must sit clearly nearer its own node than any other node, or it reads as the wrong tag's label
        if own is not None and any(box_dist(box, own) > 0.7 * box_dist(box, w) for w in H if w != own):
            return False
        return (box[0] >= frame.x0 and box[2] <= frame.x1 and box[1] >= frame.y0 and box[3] <= frame.y1
                and not any(hit(box, o, 15) for o in placed.values())  # 15 px gap: stacked labels read as one
                and not any(hit(box, nb, 0) for u, nb in nodes.items() if u != own)
                and not (edges and inside_pts(edge_px, box)))

    on_edge, worst = 0, 0.0  # worst = the most ambiguous label: own-node distance / nearest other node
    for t in sorted(H, key=lambda t: (t not in focus, -H.nodes[t]["n"], t)):
        r = np.sqrt(sizes[t]) / 2 + 2  # points
        cands = [(r, 0, "left", "center"), (-r, 0, "right", "center"), (0, r, "center", "bottom"),
                 (0, -r, "center", "top"), (r * .7, r * .7, "left", "bottom"), (-r * .7, r * .7, "right", "bottom"),
                 (r * .7, -r * .7, "left", "top"), (-r * .7, -r * .7, "right", "top")]
        done = False
        for edges in (True, False):  # prefer a spot clear of edges; never accept one on a label or node
            for dx, dy, ha, va in cands:
                txt = ax.annotate(t, pos[t], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                  fontsize=8.0, color=INK, fontweight="bold" if t in focus else "normal",
                                  path_effects=HALO)
                e = txt.get_window_extent(rend)
                if free((e.x0, e.y0, e.x1, e.y1), own=t, edges=edges):
                    placed[t] = (e.x0, e.y0, e.x1, e.y1)
                    worst = max(worst, box_dist(placed[t], t) / min(box_dist(placed[t], w) for w in H if w != t))
                    done = True
                    on_edge += not edges
                    break
                txt.remove()
            if done:
                break
        if not done:
            return fig, 99, 9.0

    # call-out: nearest spot to the recommended pair's line clear of labels, nodes and edges
    mx, my = pair_mid[best]
    for d in [8, 20, 35, 50, 70, 90, 120, 150]:
        for ang in range(0, 360, 30):
            dx, dy = d * np.cos(np.radians(ang)), d * np.sin(np.radians(ang))
            note = ax.annotate(msg, (mx, my), xytext=(dx, dy), textcoords="offset points",
                               ha="left" if dx > 1 else "right" if dx < -1 else "center",
                               va="bottom" if dy > 1 else "top" if dy < -1 else "center",
                               fontsize=7.8, color=ACCENT, fontweight="bold", path_effects=HALO,
                               arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6, shrinkA=1, shrinkB=0))
            note.update_positions(rend)
            e = plt.Text.get_window_extent(note, rend)  # text only: Annotation's extent includes the leader line
            tb = (e.x0, e.y0, e.x1, e.y1)
            lead = px(np.linspace((mx, my), ax.transData.inverted().transform(((e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2)), 60))
            lead = lead[~((lead[:, 0] > tb[0]) & (lead[:, 0] < tb[2]) & (lead[:, 1] > tb[1]) & (lead[:, 1] < tb[3]))]
            crosses = any(inside_pts(lead, o) for o in list(placed.values()) +
                          [nb for u, nb in nodes.items() if u not in best])
            if not crosses and free(tb):
                return fig, on_edge, worst
            note.remove()
    return fig, 99, 9.0


# try each layout in 12 rotations x 2 reflections. Keep only drawings where no label touches an edge and no
# edge passes under another node; of those, keep the one whose most ambiguous label is clearest.
chosen, best_worst, fig = None, None, None
for li, kk in enumerate(layouts):
    for flip in (1, -1):
        for angle in range(0, 180, 15):
            f_, bad, worst = draw(kk, angle, flip)
            if bad == 0 and (best_worst is None or worst < best_worst):
                if fig is not None:
                    plt.close(fig)
                chosen, best_worst, fig = (li, angle, flip), worst, f_
            else:
                plt.close(f_)
assert chosen, "no layout gives clear labels and clear edges"
print("layout, rotation, flip:", chosen, "worst label ratio:", round(best_worst, 2))
fig.savefig(OUT / "fig_network.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_network.svg", bbox_inches="tight", facecolor="white")
NUM.update(fig_nodes=H.number_of_nodes(), fig_edges=H.number_of_edges())
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=2, default=str))
plt.show()

# %% [markdown]
# ## 8 · Sanity checks (fail loudly)

# %%
assert "Indie" not in G, "seed tag leaked into the graph"
assert not (set(G) & set(NUM["generic"])), "generic tag leaked into the graph"
assert not (set(G) & tn.NON_DESIGN), "non-design tag leaked into the graph"
assert "Indie" not in best and not (set(best) & set(NUM["generic"])), "recommended pair is an artifact"
assert NUM["nodes"] == G.number_of_nodes() and NUM["edges"] == G.number_of_edges()
assert NUM["analysed"] >= 950, "too many games missing live data"
assert NUM["population"] > 9000, "cohort incomplete"
assert NUM["steamspy_vs_s1_jaccard"] > 0.99 and NUM["tri_top5_jaccard"] < 0.99, "S2 should differ from S1, SteamSpy should not"
assert set(best).isdisjoint(runner) or best != runner
print("all checks passed")
