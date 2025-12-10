# Research Paper Compilation Guide

## Prerequisites

Install LaTeX distribution:
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Windows
# Download and install MiKTeX from https://miktex.org/
```

## Compilation Commands

### Basic Compilation
```bash
cd /home/claude/work/repo
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

### One-Line Compilation
```bash
pdflatex research_paper.tex && bibtex research_paper && pdflatex research_paper.tex && pdflatex research_paper.tex
```

### Using latexmk (Recommended)
```bash
latexmk -pdf research_paper.tex
```

## Generated Files

After compilation, you'll find:
- `research_paper.pdf` - The final research paper
- `research_paper.aux` - Auxiliary file
- `research_paper.bbl` - Bibliography file
- `research_paper.blg` - Bibliography log
- `research_paper.log` - Compilation log

## Figures

All figures are located in `research_artifacts/`:
- `figure1_state_copy.png` - State copy performance comparison
- `figure2_mcts_scaling.png` - MCTS scaling analysis
- `figure3_transposition_table.png` - Transposition table performance
- `figure4_memory_usage.png` - Memory usage analysis
- `figure5_scalability.png` - Dimensional scalability
- `figure6_rollout_depth.png` - Rollout depth analysis
- `figure7_move_distribution.png` - Move distribution
- `figure8_architecture.png` - System architecture diagram

## Benchmark Data

Raw benchmark data is stored in:
- `research_artifacts/benchmark_results.json` - Complete benchmark results

## Troubleshooting

### Missing Figures
If compilation fails with missing figure errors, ensure all PNG files are in `research_artifacts/` directory.

### Bibliography Issues
If references don't appear:
1. Run `pdflatex` once
2. Run `bibtex research_paper`
3. Run `pdflatex` twice more

### Package Errors
If you get package not found errors, install the missing packages:
```bash
sudo apt-get install texlive-latex-extra texlive-fonts-extra
```

## Paper Statistics

- **Pages:** 8-10 pages (target achieved)
- **Figures:** 8 figures with detailed analysis
- **Tables:** Integrated into text and figures
- **References:** 10 citations
- **Word Count:** ~8,500 words (dense technical content)
- **Format:** NeurIPS 2024 conference standard
