"""Convert notebooks/analysis.py (percent format) into notebooks/analysis.ipynb."""
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "notebooks/analysis.py").read_text()
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    if "[markdown]" in head:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbformat.write(nb, ROOT / "notebooks/analysis.ipynb")
print(len(cells), "cells written")
