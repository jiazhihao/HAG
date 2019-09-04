DOC=gnn

# dependencies via include files
INCLUDED_TEX = \
  abstract.tex \
  intro.tex \
  related.tex \
  overview.tex \
  partition.tex \
  memory.tex \
  runtime.tex \
  eval.tex \
  conclusion.tex \

INCLUDED_FIGS =

$(DOC).pdf: $(DOC).tex bibliography.bib $(INCLUDED_TEX) $(INCLUDED_FIGS)
	pdflatex -t letter -halt-on-error $(DOC).tex
	bibtex $(DOC)
	pdflatex -t letter -halt-on-error $(DOC).tex
	pdflatex -t letter -halt-on-error $(DOC).tex

spelling :
	for f in *.tex; do aspell -p ./aspell.en.pws --repl=./aspell.en.prepl -c $$f; done


clean:
	rm -f *.bbl *.aux *.log *.blg *.lot *.lof *.dvi *.out $(DOC).pdf

