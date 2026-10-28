#!/usr/bin/env bash
# Build the arXiv source package for paper 1: arxiv/hermite-birkhoff-arxiv.tar.gz
#   - copies main.tex and the figures it uses into a flat directory,
#   - rewrites ../figures/ paths, drops "Draft of" from the date and the per-entry
#     verification comments from the bibliography,
#   - compiles once with pdflatex to check, and packs main.tex, main.bbl-free source and figures.
# Run from paper/:   bash make_arxiv.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT=arxiv/build
rm -rf "$OUT" && mkdir -p "$OUT/figures"
sed -e 's#\.\./figures/#figures/#g' \
    -e 's#\\date{Draft of \\today}#\\date{\\today}#' \
    -e '/^% verified/d' -e '/^% volume and pages/d' -e '/^% DOI not/d' -e '/^% from the reference/d' \
    main.tex > "$OUT/main.tex"
for f in $(grep -o 'figures/[A-Za-z0-9_]*\.pdf' "$OUT/main.tex" | sort -u); do
  cp "../$f" "$OUT/$f"
done
( cd "$OUT" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
             && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
             && grep -q "undefined" main.log && { echo "undefined references"; exit 1; } || true )
( cd "$OUT" && tar czf ../hermite-birkhoff-arxiv.tar.gz main.tex figures )
cp "$OUT/main.pdf" arxiv/hermite-birkhoff-arxiv-preview.pdf
echo "arXiv package: paper/arxiv/hermite-birkhoff-arxiv.tar.gz ($(du -h arxiv/hermite-birkhoff-arxiv.tar.gz | cut -f1))"
