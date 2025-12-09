# 5D Chess AI - Setup Instructions

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (for CuPy acceleration)
- **CUDA**: Version 11.x or 12.x
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 2GB for dependencies

### Software Dependencies
- Python 3.8+
- Node.js (for 5d-chess-js library)
- CUDA Toolkit
- pip

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` - PyTorch for neural network operations
- `cupy-cuda11x` - GPU-accelerated NumPy alternative
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualization
- `javascript` - Python-JavaScript bridge for 5d-chess-js

**Note**: If you have CUDA 12.x, install `cupy-cuda12x` instead:
```bash
pip install cupy-cuda12x
```

### 2. Install 5D Chess JavaScript Library

```bash
npm install 5d-chess-js
```

or globally:
```bash
npm install -g 5d-chess-js
```

### 3. Verify Installation

Check that all components are working:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cupy as cp; print('CuPy:', cp.__version__)"
python -c "from javascript import require; print('JavaScript bridge: OK')"
```

Expected output:
```
PyTorch: 2.x.x
CuPy: 12.x.x
JavaScript bridge: OK
```

## Configuration

### GPU Setup

Verify CUDA is available:
```bash
python -c "import cupy; print('CUDA available:', cupy.cuda.is_available())"
```

If CUDA is not available:
1. Ensure NVIDIA drivers are installed
2. Install CUDA Toolkit from NVIDIA
3. Set CUDA_PATH environment variable
4. Restart terminal/IDE

### Memory Limits

If you encounter out-of-memory errors, adjust these parameters in the code:

```python
# In Chess5D initialization
game = Chess5D(
    max_time=1,     # Reduce from 1 to handle fewer timelines
    max_turns=25    # Reduce from 25 for shorter games
)

# In MCTS configuration
config = {
    'num_searches': 10,  # Reduce from 20 for less memory usage
    'C': 1.41
}
```

## Running Tests

### Quick Validation Test
```bash
cd scripts
python quick_test.py
```
Runs a single game to verify everything works (~2-5 minutes).

### Full Performance Test
```bash
cd tests
python test_performance.py
```
Runs comprehensive performance analysis (~30-60 minutes).

### Learning Analysis
```bash
cd tests
python test_learning.py
```
Analyzes learning effectiveness (~20-40 minutes).

### Complete Test Suite
```bash
cd scripts
python run_full_analysis.py
```
Runs all tests in sequence (~1-2 hours).

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'torch'
```bash
pip install torch
```

#### 2. ModuleNotFoundError: No module named 'cupy'
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

#### 3. ImportError: cannot import name 'require' from 'javascript'
```bash
pip install javascript
npm install 5d-chess-js
```

#### 4. CUDA out of memory
Reduce parameters:
```python
max_time=1        # Use fewer timelines
max_turns=20      # Shorter games
num_searches=10   # Fewer MCTS searches
```

#### 5. JavaScript require not finding 5d-chess-js
```bash
# Install globally
npm install -g 5d-chess-js

# Or set NODE_PATH
export NODE_PATH=$(npm root -g)
```

#### 6. Slow performance without GPU
CuPy requires NVIDIA GPU. Without GPU:
- Install CPU-only version
- Replace `cupy` with `numpy` in code
- Expect 10-50x slower performance

### Performance Tips

1. **Use GPU**: Ensure CuPy is using GPU
   ```python
   import cupy as cp
   print(cp.cuda.runtime.getDeviceCount())  # Should be > 0
   ```

2. **Monitor Memory**:
   ```python
   import cupy as cp
   mempool = cp.get_default_memory_pool()
   print(f"Used: {mempool.used_bytes() / 1e9:.2f} GB")
   ```

3. **Clear Cache Periodically**:
   ```python
   cp.get_default_memory_pool().free_all_blocks()
   ```

4. **Reduce Batch Sizes**: Lower `num_searches` in MCTS config

5. **Use Optimization**: Enable OptimizedMCTS with caching

## Development Setup

### IDE Configuration

**VS Code**:
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black"
}
```

**PyCharm**:
- Enable NumPy/SciPy support
- Configure CUDA path in settings
- Set Python interpreter to conda/venv

### Directory Structure
```
.
├── src/                    # Source code
│   ├── main.py            # Original implementation
│   ├── super.py           # Enhanced MCTS implementation
│   └── jsrun.py           # JavaScript bridge utilities
├── tests/                 # Test suite
│   ├── test_performance.py
│   └── test_learning.py
├── scripts/               # Utility scripts
│   ├── quick_test.py
│   ├── optimize_architecture.py
│   ├── run_full_analysis.py
│   └── visualize_results.py
├── docs/                  # Documentation
│   ├── research_analysis.md
│   ├── RESULTS_README.md
│   └── SETUP.md
└── results/              # Test outputs (generated)
```

## Next Steps

After successful installation:

1. Run quick test to validate setup:
   ```bash
   cd scripts && python quick_test.py
   ```

2. Read the research analysis:
   ```bash
   cat docs/research_analysis.md
   ```

3. Run full test suite:
   ```bash
   cd scripts && python run_full_analysis.py
   ```

4. Generate visualizations:
   ```bash
   cd scripts && python visualize_results.py
   ```

5. Review results:
   ```bash
   ls -lh results/
   ```

## Support

For issues or questions:
1. Check this setup guide
2. Review `docs/research_analysis.md`
3. Examine error logs in test output
4. Verify CUDA and GPU are working
5. Ensure all dependencies are installed

## References

- PyTorch: https://pytorch.org/get-started/locally/
- CuPy: https://docs.cupy.dev/en/stable/install.html
- 5d-chess-js: https://www.npmjs.com/package/5d-chess-js
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

---

Last Updated: 2025-12-09
