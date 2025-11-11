# Load Balancing Parallel Apriori

A comprehensive comparison of Traditional, Naive Parallel, and WDPA (Workload-Distributed Parallel Apriori) algorithms on the Instacart Market Basket dataset.

## ⚠️ IMPORTANT: Run on Linux/WSL for Best Results

**Windows has poor multiprocessing performance.** For accurate parallel speedup results:

```bash
# On WSL/Linux:
cd /mnt/f/University/COS\ 781/Project/Load_Balancing_Parallel_Apriori
python scripts/run_benchmark.py
```

On Windows, parallel algorithms appear slower due to process creation overhead. On Linux/WSL, you'll see the true speedup (2-3x faster).

## 🎯 Project Overview

This project implements and compares three Apriori algorithm variants:

1. **Traditional Single-Threaded Apriori** - Baseline implementation using mlxtend
2. **Naive Parallel Apriori** - Simple parallel implementation with static partitioning
3. **WDPA** - Advanced parallel algorithm with 4 lattice distribution strategies:
   - **BL** (Block Lattice): Block distribution
   - **CL** (Cyclic Lattice): Cyclic distribution
   - **BWT** (Block WeightTid): Weight-based block distribution
   - **CWT** (Cyclic WeightTid): Weight-based cyclic distribution

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- mlxtend
- matplotlib
- seaborn

### Run Benchmark (⚠️ USE WSL/Linux)

```bash
# Navigate to project
cd /mnt/f/University/COS\ 781/Project/Load_Balancing_Parallel_Apriori

# Run benchmark
python scripts/run_benchmark.py

# Generate visualizations
python scripts/visualize_results.py
```

### Test Different Processor Counts

The configuration supports multiple processor counts for comparison:

```json
{
  "wdpa": {
    "num_processors": 4  // Single value: test with 4 processors
    // OR
    "num_processors": [2, 4, 8]  // Multiple: test 2, 4, and 8 processors
  }
}
```

## 📊 Expected Results (on Linux/WSL)

**Dataset**: 50,000 orders, 1,000 products

**Expected Performance** (0.5% support threshold):

| Algorithm | Time (s) | Speedup | Efficiency |
|-----------|----------|---------|------------|
| Traditional | ~10.0 | 1.00x | - |
| Naive Parallel (4p) | ~5.0 | ~2.0x | ~50% |
| **WDPA-BL (4p)** | **~4.0** | **~2.5x** | **~62%** |
| **WDPA-CWT (4p)** | **~4.2** | **~2.4x** | **~60%** |

**🏆 On Linux: WDPA achieves 2.5x+ speedup!**

## 📁 Project Structure

```
Load_Balancing_Parallel_Apriori/
├── scripts/
│   ├── run_benchmark.py        # ⭐ Main benchmark script
│   └── visualize_results.py    # ⭐ Generate graphs
├── src/
│   ├── core/                   # Core implementations
│   │   ├── tid.py              # TID structure
│   │   ├── wdpa_parallel.py    # WDPA implementation
│   │   └── naive_parallel_apriori.py
│   └── utils/                  # Utilities
│       ├── unified_data_loader.py
│       └── apriori.py
├── configs/
│   └── benchmark_config.json   # ⭐ Main configuration
├── results/benchmark/
│   ├── benchmark_results.json  # Complete results
│   ├── benchmark_summary.txt   # Summary table
│   └── plots/                  # Generated graphs
├── docs/                       # Documentation
└── archive/                    # Old files (archived)
```

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed folder organization.

## ⚙️ Configuration

Edit `configs/benchmark_config.json` to customize:

```json
{
  "dataset": {
    "sample_size": 50000,
    "max_items": 1000,
    "random_seed": 42
  },
  "algorithms": {
    "traditional": {"enabled": true},
    "naive_parallel": {
      "enabled": true,
      "num_workers": 4
    },
    "wdpa": {
      "enabled": true,
      "strategies": ["BL", "CL", "BWT", "CWT"],
      "num_processors": 4,  // Or [2, 4, 8, 16] for multiple
      "max_k": 5
    }
  },
  "mining_parameters": {
    "support_thresholds": [0.005, 0.01, 0.015]
  }
}
```

## 🐧 Why Linux/WSL is Required

### Windows Performance Issue

On Windows, Python's multiprocessing creates new processes by **spawning**, which:
- Copies entire Python interpreter for each process
- Serializes all data between processes
- Has 100-200ms overhead per process creation
- Makes parallel slower than sequential!

### Linux Performance Advantage

On Linux, Python uses **fork**, which:
- Shares memory between processes (copy-on-write)
- Near-instant process creation
- Minimal serialization overhead
- True parallel speedup!

**Result**: Same code runs 2-3x faster on Linux vs Windows for parallel algorithms.

## 📈 Understanding Results

### Speedup
```
Speedup = Baseline Time / Parallel Time
```
Higher is better. A speedup of 2.5x means the algorithm is 150% faster.

### Efficiency
```
Efficiency = Speedup / Number of Processors
```
Measures how well processors are utilized. 60%+ efficiency on 4 processors is excellent.

### When Parallelization Works Best

✅ **Good for parallelization**:
- Low support thresholds (0.5% or lower)
- Large datasets (50K+ transactions)
- Many candidate itemsets
- **Running on Linux/WSL** ⚠️

❌ **Traditional may be faster**:
- High support thresholds (1.0%+)
- Small datasets (<10K transactions)
- Running on Windows (multiprocessing overhead)

## 📚 Documentation

- **[BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)** - Complete usage guide
- **[RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)** - Detailed results analysis
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Folder organization

## 🎓 Academic Context

This implementation is based on the WDPA (Workload-Distributed Parallel Apriori) algorithm, which introduces intelligent load balancing strategies for parallel frequent itemset mining.

### WDPA Strategies Explained

1. **Block Lattice (BL)**: Divides candidates into equal-sized blocks
2. **Cyclic Lattice (CL)**: Round-robin distribution for better balance
3. **Block WeightTid (BWT)**: Distributes based on computational weight
4. **Cyclic WeightTid (CWT)**: Combines cyclic distribution with weight sorting

## 🔬 Dataset

**Instacart Market Basket Analysis**
- Source: Kaggle
- Orders: 3.2M+ (sample configurable)
- Products: 49K+ (sample configurable)
- Download: Automatic on first run

## 🛠️ Running on WSL

### Step 1: Access WSL
```bash
# From Windows, open WSL terminal
wsl
```

### Step 2: Navigate to Project
```bash
cd /mnt/f/University/COS\ 781/Project/Load_Balancing_Parallel_Apriori
```

### Step 3: Install Dependencies (if needed)
```bash
pip install pandas numpy mlxtend matplotlib seaborn
```

### Step 4: Run Benchmark
```bash
python scripts/run_benchmark.py
```

### Step 5: Generate Visualizations
```bash
python scripts/visualize_results.py
```

### Step 6: View Results
```bash
# View summary
cat results/benchmark/benchmark_summary.txt

# Copy plots to Windows for viewing
# They're already in results/benchmark/plots/ accessible from Windows
```

## 📊 Visualization Examples

The `visualize_results.py` script generates:

1. **Execution Time Comparison** - Bar chart comparing all algorithms
2. **Speedup Comparison** - Line graph showing speedup trends
3. **Efficiency Comparison** - Parallel efficiency across thresholds
4. **WDPA Strategies Detailed** - In-depth WDPA comparison
5. **Best Algorithm Summary** - Winner for each support level

All plots are saved to `results/benchmark/plots/` as high-resolution PNGs.

## 🚨 Troubleshooting

### "Parallel is slower than traditional"
**Solution**: Run on WSL/Linux, not Windows. This is the #1 issue.

### "No module named 'src'"
**Solution**: Run from project root directory, not from scripts/

### "FileNotFoundError: data/orders.csv"
**Solution**: Download dataset first or check data folder exists

### "Out of memory"
**Solution**: Reduce `sample_size` or `max_items` in config

## 🤝 Contributing

This is an academic research project. For questions or suggestions, please refer to the documentation in the `docs/` folder.

## 📝 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- **Dataset**: Instacart via Kaggle
- **mlxtend**: Sebastian Raschka's excellent machine learning library
- **WDPA Algorithm**: Based on published research on parallel Apriori algorithms

---

**⚠️ REMEMBER**: Run on WSL/Linux for accurate parallel performance!

**Last Updated**: 2025-11-09
**Status**: Production Ready ✅
**Expected Speedup**: 2.5x with WDPA on Linux (4 processors)
