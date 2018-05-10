BUILD := paper

DEPS := introduction.tex related.tex approach.tex evaluation.tex conclusion.tex

LATEX  := pdflatex
BIBTEX := bibtex

TARGETS := $(patsubst %, %.pdf, $(BUILD))

# phony targets

all: $(TARGETS)

clean:
	rm -rf *.aux *.log *.blg *.bbl *.ent *.out *.dvi *.ps *.pdf *.tar.gz

arxiv: paper.pdf # just build the paper because we want to build the .bbl
	tar czvf paper.tar.gz paper.tex paper.bbl $(DEPS)

.PHONY: all clean arxiv

# main targets

%.pdf: %.tex %.bib $(DEPS)
	$(eval SRC_$@ = $(patsubst %.tex, %, $<))
	$(LATEX) $(SRC_$@)
	$(BIBTEX) $(SRC_$@)
	$(LATEX) $(SRC_$@)
	$(LATEX) $(SRC_$@)
