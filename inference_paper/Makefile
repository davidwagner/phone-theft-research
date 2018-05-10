BUILD := \
	paper


DEPS := \
	algorithm.sty \
	algorithmic.sty \
	icml2018.bst \
	icml2018.sty \
	fancyhdr.sty \
	macros.tex \
	abstract.tex \
	introduction.tex \
	background.tex \
	discussion.tex \
	techniques.tex \
	breaks.tex \
	obfuscated-gradients.tex \
	conclusion.tex \
	acknowledgements.tex \
	bibliography.tex \
	figures/defense_gan.png \
	figures/distortion.png \
	figures/lid.pdf \
	figures/sap.pdf \
	figures/therm.pdf \
	figures/thermloss.pdf \



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
