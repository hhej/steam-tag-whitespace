"""Render the one-page report: outputs/numbers.json + report/template.html -> report/report.html -> report.pdf.

Every number on the page comes from outputs/numbers.json (written by notebooks/analysis.ipynb).
Fails loudly if the PDF is not exactly one A4 page.
"""
import json
import re
import math
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/hhej/steam-tag-whitespace"
TITLE = "Which tag pair should an indie game use? {A} × {B}"

N = json.loads((ROOT / "outputs/numbers.json").read_text())
A, B = N["best"]
sweep = {r["param"]: r["rank"] for r in N["sweep"]}


def rank(v):
    return "not eligible" if v is None or (isinstance(v, float) and math.isnan(v)) else f"rank #{int(v)}"


def pct(x):
    return f"{100 * x:.0f}%"


def row(r, role, cls="", h=None):
    return (f'<tr class="{cls}"><td><span class="role">{role}:</span> {r["pair"][0]} × {r["pair"][1]}</td><td>{r["n"]}</td><td>{r["expected"]:.0f}</td>'
            f'<td>{r["sat"]:.2f}×</td><td>{pct(r["overlap"])}</td>'
            f'<td>{r["median"]:.0f} ({r["lift"]:.1f}×)</td></tr>')


pk, pb, cr = N["pick"], N["planb"], N["crowd"]
V = dict(
    repo=REPO, repo_short=REPO.replace("https://", ""), A=A, B=B, PA=pb["pair"][0], PB=pb["pair"][1],
    CA=cr["pair"][0], CB=cr["pair"][1],
    fetched=N["fetched_from"] if N["fetched_from"] == N["fetched_to"] else f"{N['fetched_from']}–{N['fetched_to']}",
    pop=f"{N['population']:,}", an=f"{N['analysed']:,}", jac=f"{N['tri_top5_jaccard']:.2f}",
    rho=f"{N['tri_reviews_spearman']:.2f}",
    generic=", ".join(N["generic"]), raw_touch=N["raw_top10_touch_common"], npmi_touch=N["npmi_top10_touch_common"],
    k20_n=N["k20_n"], k20_lift=f"{N['k20_lift']:.1f}",
    min_tag=N["min_tag"], min_co=N["min_co"], min_pair=N["min_pair"],
    tags_before=N["tags_before"], nodes=N["nodes"], edges=N["edges"], fig_nodes=N["fig_nodes"],
    n_ab=pk["n"], exp=f"{pk['expected']:.0f}", sat=f"{pk['sat']:.2f}", ov=pct(pk["overlap"]), med=f"{pk['median']:.0f}",
    lift=f"{pk['lift']:.1f}", cmed=f"{N['median_reviews']:.0f}",
    perm="p &lt; 0.001" if N["perm_p"] < 0.001 else f"p = {N['perm_p']:.3f}",
    hit=pct(pk["hit"]), hit_cut=f"{N['hit_cut']:,.0f}", clift=f"{cr['lift']:.1f}", csat=f"{cr['sat']:.2f}",
    top10_crowded=N["top10_lift_crowded"], cut_n=sum(1 for k in sweep if k.startswith("NPMI")),
    boot_gt1=N["boot_lift_gt1"], lo=f"{N['boot_lift_p05']:.1f}", hi=f"{N['boot_lift_p95']:.1f}", top1=N["boot_top1"],
    floor=N["boot_below_floor"], over=N["boot_sat_over"], s2n=N["s2_pair_n"], s2lift=f"{N['s2_pair_lift']:.1f}",
    rows=row(pk, "#1", "pick", h=N["holdout"]["pick"]) + row(pb, "#2", "planb", h=N["holdout"]["planb"])
         + row(N["third_rec"], "#3", "third", h=N["holdout"]["third"]) + row(cr, "Crowded (box)", h=N["holdout"]["crowd"]),
    hn=f"{N['holdout_n']:,}", hsat=f"{N['holdout']['pick']['sat']:.2f}", hlift=f"{N['holdout']['pick']['lift']:.1f}",
    h3lift=f"{N['holdout']['third']['lift']:.1f}", fp25=pct(N["holdout_fp_share"]),
    bridge=N["bridge"][0], hops=N["bridge_hops"], hops_wo=N["bridge_hops_without"],
    wo5=f"{N['pick_lift_wo_top5']:.2f}", unc=N["uncrowded_n"], ovr=N["overlap_removed"],
    ova=N["overlap_removed_eg"][0], ovb=N["overlap_removed_eg"][1], ovp=pct(N["overlap_removed_eg_ov"]),
    npmi=f"{pk['npmi']:.2f}", pbn=pb["n"], pbexp=f"{pb['expected']:.0f}", pblift=f"{pb['lift']:.1f}",
    pbnpmi=f"{pb['npmi']:.2f}", pbsat=f"{pb['sat']:.2f}", top3=N["boot_top3"],
    coop=pct(N["unmerged_hm_coop"] / N["unmerged_hm_n"]), horror=pct(N["pick_horror"] / pk["n"]),
    top_game=N["pick_top_game"], TA=N["third"][0], TB=N["third"][1],
    col_lo=f"{N['colour_lo']:.0f}", col_hi=f"{N['colour_hi']:.0f}",
    hl=f"{N['horror_lift']:.1f}", nhl=f"{N['pick_nonhorror_lift']:.1f}",
    h2sat=f"{N['holdout']['planb']['sat']:.2f}",
    cnpmi=f"{cr['npmi']:.2f}",
    t3n=N["third_rec"]["n"], t3exp=f"{N['third_rec']['expected']:.0f}", t3lift=f"{N['third_rec']['lift']:.1f}",
)
assert not pk["edge"] and not pb["edge"], "page says both pairs have no edge"
V["title"] = TITLE.format(**V)

html = (ROOT / "report/template.html").read_text().format_map(V)
(ROOT / "report/report.html").write_text(html)

pdf = ROOT / "report/6810424009-PanjapolAmpornratana-MADT7202-MidTerm.pdf"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                f"--print-to-pdf={pdf}", (ROOT / "report/report.html").as_uri()], check=True, capture_output=True)
pages = len(PdfReader(pdf).pages)
print(f"wrote {pdf}: {pages} page(s)")
assert pages == 1, "report overflows one page"

# Chrome silently shrinks a page whose content is wider than A4, which would push 10pt text below 10pt.
# 10pt Helvetica gives word boxes ~11.6pt tall; a shrunken page gives ~11.1pt.
bbox = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
heights = sorted(float(m[1]) - float(m[0]) for m in re.findall(r'yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)"', bbox))
body = max(set(round(h, 1) for h in heights), key=[round(h, 1) for h in heights].count)
print(f"body text box height {body}pt (10pt unscaled = 11.6)")
assert body >= 11.5, "Chrome shrank the page to fit: some element is wider than the page, so text is below 10pt"
