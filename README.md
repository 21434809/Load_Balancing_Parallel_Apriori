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

## 📝 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- **Dataset**: Instacart via Kaggle
- **mlxtend**: Sebastian Raschka's excellent machine learning library
- **WDPA Algorithm**: Based on published research on parallel Apriori algorithms

---
