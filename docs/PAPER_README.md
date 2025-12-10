# Research Paper - Enhanced MCTS for 5D Chess

## Compilation Instructions

### Option 1: Local LaTeX Installation

If you have LaTeX installed locally:

```bash
cd docs/latex
make
```

Or manually:

```bash
cd docs/latex
pdflatex research_paper.tex
pdflatex research_paper.tex
pdflatex research_paper.tex
```

(Run three times to resolve all references)

### Option 2: Docker

If you don't have LaTeX installed:

```bash
cd docs/latex
make docker
```

### Option 3: Online LaTeX Editor

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project
3. Upload `research_paper.tex`
4. Upload all figures from `docs/figures/`
5. Compile

## Paper Structure

The paper includes:

- **Abstract**: Summary of contributions and results
- **Introduction**: Motivation and problem statement
- **Background**: MCTS and 5D chess complexity analysis
- **Architecture**: Enhanced MCTS with progressive widening, RAVE, virtual loss
- **Methodology**: Experimental setup and metrics
- **Results**: Comprehensive performance analysis with 6 figures
- **Discussion**: Insights, limitations, and future work
- **Conclusion**: Summary and impact

## Figures Included

All figures are located in `docs/figures/`:

1. `win_rates.png` - Win rates vs MCTS simulation count
2. `game_lengths.png` - Average game length vs search depth
3. `performance_scaling.png` - Move time and node expansion scaling
4. `learning_curves.png` - Rolling win rates over game sequence
5. `timeline_expansions.png` - Timeline management efficiency
6. `comprehensive_analysis.png` - Multi-metric comparison

## Key Results

- **Win Rate**: 62% for strongest configuration (800 simulations)
- **Game Quality**: 88% increase in game length (19.9 to 37.4 moves)
- **Efficiency**: Linear scaling, 892 nodes/second average
- **Timeline Management**: 49% reduction in boundary violations
- **Improvement**: 18% win rate increase over baseline

## Citation

If you use this work, please cite:

```bibtex
@article{5dchess2025,
  title={Enhanced Monte Carlo Tree Search for 5D Chess: Architecture Optimization and Performance Analysis},
  author={Research Team},
  journal={arXiv preprint},
  year={2025}
}
```

## Requirements for Compilation

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Packages: amsmath, graphicx, booktabs, hyperref, algorithm, algorithmic
- All required packages are included in standard LaTeX distributions

## Troubleshooting

### Missing Packages

If you get "package not found" errors:

```bash
# For Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-recommended

# For macOS with Homebrew
brew install --cask mactex

# For Windows
# Download and install MiKTeX from https://miktex.org/
```

### Figure Not Found

Make sure all PNG files are in the correct location relative to the .tex file:

```
docs/
├── latex/
│   └── research_paper.tex
└── figures/
    ├── win_rates.png
    ├── game_lengths.png
    ├── performance_scaling.png
    ├── learning_curves.png
    ├── timeline_expansions.png
    └── comprehensive_analysis.png
```

### Compilation Errors

1. Check that all figures exist
2. Ensure LaTeX packages are installed
3. Try compiling multiple times (references may need multiple passes)
4. Check the `.log` file for specific errors

## Contact

For questions or issues, please open an issue in the repository.
