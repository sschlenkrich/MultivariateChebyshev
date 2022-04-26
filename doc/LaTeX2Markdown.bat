pandoc MultivariateChebyshev.tex -s ^
  --filter pandoc-crossref ^
  --citeproc ^
  --bibliography=.\refs\References.bib ^
  --mathjax ^
  --toc ^
  -o MultivariateChebyshev.md
pause
