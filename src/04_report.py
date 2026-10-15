"""Render the one-page report: outputs/numbers.json + report/template.html -> report/report.html -> report.pdf.

Every number on the page comes from outputs/numbers.json (written by notebooks/analysis.ipynb).
Fails loudly if the PDF is not exactly one A4 page.
"""
import json
import math
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/hhej/steam-tag-whitespace"
TITLE = "Which Steam tag pair should an indie studio build next? {A} × {B}"

N = json.loads((ROOT / "outputs/numbers.json").read_text())
A, B = N["best"]
sweep = {r["param"]: r["rank"] for r in N["sweep"]}


def rank(v):
    return "not eligible" if v is None or (isinstance(v, float) and math.isnan(v)) else f"rank #{int(v)}"


V = dict(
    repo=REPO, repo_short=REPO.replace("https://", ""), A=A, B=B, RA=N["runner"][0], RB=N["runner"][1],
    FA=N["fallback"][0], FB=N["fallback"][1],
    fetched=N["fetched_from"] if N["fetched_from"] == N["fetched_to"] else f"{N['fetched_from']}–{N['fetched_to']}",
    pop=f"{N['population']:,}", an=f"{N['analysed']:,}", jac=f"{N['tri_top5_jaccard']:.2f}",
    rho=f"{N['tri_reviews_spearman']:.2f}", ssjac=f"{N['steamspy_vs_s1_jaccard']:.2f}",
    generic=", ".join(N["generic"]), raw_touch=N["raw_top10_touch_common"], npmi_touch=N["npmi_top10_touch_common"],
    raw_edge="–".join(N["raw_top_edge"]), raw_edge_n=N["raw_top_edge_count"], k20_n=N["k20_n"], k20_lift=f"{N['k20_lift']:.1f}",
    tags_before=N["tags_before"], nodes=N["nodes"], edges=N["edges"], fig_nodes=N["fig_nodes"],
    broker=N["top_broker"], bc=f"{N['top_broker_bc']:.2f}",
    broker_deg=N["top_broker_degree"],
    n_ab=N["best_n"], exp=f"{N['best_expected']:.0f}", ov=f"{N['best_overlap']:.2f}", med=f"{N['best_median']:.0f}",
    lift=f"{N['best_lift']:.1f}", cmed=f"{N['median_reviews']:.0f}", perm=f"{N['perm_p']:.3f}",
    flift=f"{N['fallback_lift']:.1f}", fsat=f"{N['fallback_sat']:.1f}",
    cut_n=sum(1 for k in sweep if k.startswith("NPMI")),
    boot_gt1=N["boot_lift_gt1"], lo=f"{N['boot_lift_p05']:.1f}", hi=f"{N['boot_lift_p95']:.0f}", top1=N["boot_top1"],
)
cut_ok = sum(1 for k, v in sweep.items() if k.startswith("NPMI") and v == 1)
assert cut_ok == V["cut_n"], f"pair is #1 at only {cut_ok}/{V['cut_n']} cutoffs - fix the robustness wording"
V["title"] = TITLE.format(**V)

html = (ROOT / "report/template.html").read_text().format_map(V)
(ROOT / "report/report.html").write_text(html)

pdf = ROOT / "report/report.pdf"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                f"--print-to-pdf={pdf}", (ROOT / "report/report.html").as_uri()], check=True, capture_output=True)
pages = len(PdfReader(pdf).pages)
print(f"wrote {pdf}: {pages} page(s)")
assert pages == 1, "report overflows one page"
