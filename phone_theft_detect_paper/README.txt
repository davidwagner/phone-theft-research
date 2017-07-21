This zip file contains all of the necessary templates to create a
SOUPS 2017 conference paper in either LaTeX or Microsoft Word Format.

The SOUPS 2017 conference format is the ACM SIG Proceedings Template
Option 2 (Tighter Alternate style). You can download the original
templates from the ACM website, at:
http://www.acm.org/sigs/publications/proceedings-templates

The templates in this directory have been modified to include the SOUPS
2017 conference information.


***************************
*** Note to LaTeX Users ***
***************************

If you use LaTeX to create your paper, SOUPS 2017 recommends that you
use pdflatex directly. 

SOUPS 2017 specifically recommends against using latex to create a DVI
file and then convert this file to a PS and then convert the PS to a
PDF. The problem with this process is that the process sometimes
includes the TeX bitmap fonts rather than the PDF fonts, resulting in
a PDF file that can be printed but which is very difficult to read on
a screen.

Using pdflatex instead of latex should not present a problem for SOUPS
2016 papers. There are just two significant areas where pdflatex is
not compatible with LaTeX:

    1. pdflatex cannot import PostScript (.ps) or Encapsulated
    PostScript (.eps) illustrations. They must be converted to PDF
    illustrations first. You can do this with the "eps2pdf" and
    "ps2pdf" commands.  


    2. You cannot use the "pstricks" LaTeX package, which means that
    you can't create TeX illustrations such as text that follows a
    spiral path. (This shouldn't be a problem for most SOUPS papers
    either.)

