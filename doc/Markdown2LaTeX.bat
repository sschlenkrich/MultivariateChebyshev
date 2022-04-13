pandoc MultivariateChebyshev.md -s ^
  --filter pandoc-crossref ^
  --citeproc ^
  --bibliography=.\refs\References.bib ^
  --mathjax ^
  --toc ^
  -o MultivariateChebyshev.tex
pause
