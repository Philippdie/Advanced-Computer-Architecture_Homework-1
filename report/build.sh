#!/usr/bin/env sh

set -eu

cd "$(dirname "$0")"

PATH="/Library/TeX/texbin:$PATH"

pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
