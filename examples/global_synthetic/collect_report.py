"""Copy ONLY the finished HTML report to ./report (no matchups, results, notebooks or data).

usage: python collect_report.py <directory validate.py was run in>
"""

import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = sys.argv[1] if len(sys.argv) > 1 else "."
SRC = os.path.join(WORK, "oceanval_report", "_build", "html", "notebooks")
OUT = os.path.join(HERE, "report")

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
for path in sorted(glob.glob(os.path.join(SRC, "*"))):
    if path.endswith((".html", ".svg")):
        shutil.copy(path, OUT)
summary = sorted(glob.glob(os.path.join(OUT, "*_summary.html")))
assert summary, f"no report pages found in {SRC}"
with open(os.path.join(OUT, "index.html"), "w") as index:
    name = os.path.basename(summary[0])
    index.write(f'<!DOCTYPE html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={name}"><a href="{name}">Open the report</a>\n')
print(f"{len(os.listdir(OUT))} files, {sum(os.path.getsize(p) for p in glob.glob(OUT + '/*')) / 1e6:.1f} MB in {OUT}")
