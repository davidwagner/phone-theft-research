ALL = paper.pdf

all: $(ALL)

.SUFFIXES: .pic .tex .ps .dvi .bib .bbl .aux .pdf

%.pdf: %.tex
	pdflatex $<
	@echo Checking whether we should rerun LaTeX or not...
	@(grep Rerun $(<:.tex=.log) >/dev/null && pdflatex $<; exit 0)

%.bbl: %.bib %.tex
	pdflatex $(<:.bib=)
	@echo Building bibliography...
	bibtex $(<:.bib=)

clean:
	-rm -f *.{aux,dvi,log,bbl,blg}

veryclean: clean
	-rm -f $(ALL)
