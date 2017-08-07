release:
	make clean
	make soups-article.pdf

zip:
	/bin/rm -f *.aux *.bbl *.blg *.log *~
	(cd ..;zip latex-templates.zip latex-templates/*)

soups-article.pdf: soups-article.tex sigproc.bib
	pdflatex soups-article
	bibtex soups-article
	pdflatex soups-article
	pdflatex soups-article

clean:
	/bin/rm -f soups.{aux,bbl,blg,log,pdf}
	/bin/rm -f soups-article.{aux,bbl,blg,log,pdf}
