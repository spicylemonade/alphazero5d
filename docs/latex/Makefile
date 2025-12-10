# Makefile for compiling the research paper

PAPER = research_paper
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex
	$(LATEX) $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).bbl $(PAPER).blg

distclean: clean
	rm -f $(PAPER).pdf

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf || open $(PAPER).pdf

# Alternative compilation with docker if LaTeX not installed locally
docker:
	docker run --rm -v $(PWD):/data -w /data texlive/texlive:latest pdflatex $(PAPER).tex
	docker run --rm -v $(PWD):/data -w /data texlive/texlive:latest pdflatex $(PAPER).tex

help:
	@echo "Available targets:"
	@echo "  all     - Compile the paper (default)"
	@echo "  clean   - Remove auxiliary files"
	@echo "  distclean - Remove all generated files including PDF"
	@echo "  view    - Open the PDF"
	@echo "  docker  - Compile using Docker (if LaTeX not installed)"
