# Traditional Single-Threaded Apriori Algorithm

A data mining project implementing the traditional single-threaded Apriori algorithm for association rule mining on the Instacart Market Basket Analysis dataset.

## Project Overview

This project implements the traditional single-threaded Apriori algorithm for frequent itemset mining and association rule generation. It serves as the baseline implementation for comparison with parallel and load-balanced versions in the broader COS 781 research project.

## Features

- Traditional single-threaded Apriori algorithm implementation
- Support for configurable minimum support thresholds (0.15–0.3% expressed as 0.0015–0.003)
- JSON output for results analysis
- Efficient handling of large datasets (3.4M+ transactions)
- Association rule mining with lift, confidence, and support metrics

## Dataset

The project uses the Instacart Market Basket Analysis dataset:
- **Scale**: 3.4+ million grocery purchases from 200,000 shoppers
- **Products**: 50,000+ unique items across 32 million interactions
- **Sparsity**: ~10 products per order (typical for retail data)
- **Files**: orders.csv, products.csv, order_products__prior.csv, etc.

### Download dataset automatically
Use the helper script to download and place the files into `data/`.

```bash
# Ensure your virtual environment is active

# Install KaggleHub (one-time)
pip install kagglehub

# Run from the src directory so files are placed in ../data
cd src
python download_dataset.py
cd ..
```

Notes:
- If authentication is required, configure Kaggle API credentials (see `https://www.kaggle.com/docs/api`).
- After the script completes, verify `data/` contains CSVs like `orders.csv`, `products.csv`, `order_products__prior.csv`.

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Load_Balancing_Parallel_Apriori
```

2. (Recommended) Create and activate a virtual environment:
```bash
# Windows PowerShell
python -m venv venv
./venv/Scripts/Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the Instacart dataset is in the `data/` directory

## Usage

### Main Analysis
Run the complete analysis with JSON output:
```bash
python main_traditional.py
```

## Project Structure

```
Load_Balancing_Parallel_Apriori/
├── main_traditional.py    # Baseline traditional Apriori execution script
├── src/
│   └── apriori.py         # Traditional Apriori implementation
├── data/                  # Instacart dataset files
│   ├── orders.csv
│   ├── products.csv
│   ├── order_products__prior.csv
│   └── ...
├── results/               # JSON output files (generated)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Output

The analysis generates JSON files in the `results/` directory:
- `complete_analysis.json` - Complete results with all data
- `analysis_summary.json` - Summary statistics
- `apriori_results_support_*.json` - Individual results per support threshold

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- mlxtend >= 0.22.0

## Academic Context

This project is part of COS 781 Data Mining coursework (Group 16), focusing on:
- Traditional Apriori algorithm implementation
- Baseline performance measurement
- Association rule mining on large-scale retail data
- Foundation for parallel algorithm comparison

## Group Members

- Rueben van der Westhuize (u21434809)
- Kenneth Collis (u23897300)
- Marcel le Roux (u22598805)
- Stefan Tolken (u22525778)
