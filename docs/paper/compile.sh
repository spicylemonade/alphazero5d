#!/bin/bash
# Compile LaTeX document

cd /home/claude/work/repo/docs/paper

# Run pdflatex twice for references
pdflatex -interaction=nonstopmode research_paper.tex
pdflatex -interaction=nonstopmode research_paper.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg

echo "PDF generated: research_paper.pdf"
