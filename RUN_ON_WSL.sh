#!/bin/bash
# WSL Run Script for Load Balancing Parallel Apriori Benchmark
#
# This script runs the benchmark on WSL/Linux for accurate parallel performance
#
# Usage:
#   From Windows: wsl ./RUN_ON_WSL.sh
#   From WSL: ./RUN_ON_WSL.sh

echo "================================"
echo "WDPA Benchmark - WSL/Linux Mode"
echo "================================"
echo ""
echo "⚠️  Running on WSL for TRUE parallel speedup!"
echo "   (Windows multiprocessing has ~100-200ms overhead per process)"
echo ""

# Get the script directory (works in WSL)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null
then
    echo "❌ Python not found! Please install Python 3."
    exit 1
fi

# Use python3 or python
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Using Python: $PYTHON"
echo ""

# Check dependencies
echo "Checking dependencies..."
$PYTHON -c "import pandas, numpy, mlxtend, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    $PYTHON -m pip install pandas numpy mlxtend matplotlib seaborn
    echo ""
fi

echo "✅ Dependencies OK"
echo ""

# Run benchmark
echo "================================"
echo "Running Benchmark..."
echo "================================"
echo ""

$PYTHON scripts/run_benchmark.py

# Check if benchmark succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ Benchmark Complete!"
    echo "================================"
    echo ""

    # Show summary if it exists
    if [ -f "results/benchmark/benchmark_summary.txt" ]; then
        echo "📊 Quick Summary:"
        echo "----------------"
        head -n 50 results/benchmark/benchmark_summary.txt
        echo ""
    fi

    echo "📁 Results saved to:"
    echo "   - results/benchmark/benchmark_results.json"
    echo "   - results/benchmark/benchmark_summary.txt"
    echo ""

    # Ask about visualization
    read -p "Generate visualizations? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Generating plots..."
        $PYTHON scripts/visualize_results.py

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Plots generated in: results/benchmark/plots/"
            echo "   (Accessible from Windows at: F:\\University\\COS 781\\Project\\Load_Balancing_Parallel_Apriori\\results\\benchmark\\plots\\)"
        fi
    fi
else
    echo ""
    echo "❌ Benchmark failed! Check error messages above."
    exit 1
fi

echo ""
echo "================================"
echo "✅ ALL DONE!"
echo "================================"
echo ""
echo "View results:"
echo "  cat results/benchmark/benchmark_summary.txt"
echo ""
echo "Or in Windows:"
echo "  F:\\University\\COS 781\\Project\\Load_Balancing_Parallel_Apriori\\results\\benchmark\\"
echo ""
