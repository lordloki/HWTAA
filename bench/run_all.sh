#!/bin/sh
# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Reproduces every table and figure of the paper (about 25 minutes on 2 CPU cores; any OpenGL 4.3 driver, or Mesa llvmpipe).
set -e
cd "$(dirname "$0")"
# python3 on Linux; python on Windows (Git Bash), where python3 is usually the Store stub.
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if python3 -c "" 2>/dev/null; then PY=python3; else PY=python; fi
fi
"$PY" hwtaa_bench.py quality
"$PY" hwtaa_bench.py ablations
"$PY" hwtaa_bench.py window
"$PY" hwtaa_bench.py ghost
"$PY" hwtaa_bench.py frames
"$PY" make_tables.py
"$PY" make_figures.py en
"$PY" make_figures.py es
cd ../en && latexmk -pdf -interaction=nonstopmode hwtaa.tex
cd ../es && latexmk -pdf -interaction=nonstopmode hwtaa_es.tex
