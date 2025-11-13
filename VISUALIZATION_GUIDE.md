# Enhanced Visualization Guide

## Overview

The `visualize_results_enhanced.py` script creates publication-quality graphs with advanced filtering and analytics capabilities for your research paper.

---

## Quick Start

### Basic Usage
```bash
python visualize_results_enhanced.py
```

---

## Configuration Options

### 1. Filter Processor Counts

**Location:** Lines 25-28

```python
# Show only specific processor counts
PROCESSORS_TO_SHOW = [1, 4, 8]  # Only show 1, 4, and 8 processor results

# OR show all available processors
PROCESSORS_TO_SHOW = None  # Show everything
```

**Example Use Cases:**
- `[1, 4, 8]` - Compare baseline, small, and medium parallelization
- `[2, 4, 8, 16]` - Show power-of-2 scaling
- `[1, 8]` - Compare only baseline and maximum parallelization
- `None` - Include all available processor counts in your data

---

### 2. Filter Support Thresholds

**Location:** Lines 30-33

```python
# Show only specific support thresholds
SUPPORT_THRESHOLDS_TO_SHOW = [0.0008, 0.001, 0.002]  # Specific thresholds

# OR show all available thresholds
SUPPORT_THRESHOLDS_TO_SHOW = None  # Show everything
```

**Example Use Cases:**
- `[0.0008, 0.001, 0.002]` - Focus on low-support scenarios (more candidates)
- `[0.001, 0.003]` - Compare extremes (most vs fewer candidates)
- `[0.001]` - Single threshold deep-dive analysis
- `None` - Include all thresholds from your benchmark

---

### 3. Publication Mode Settings

**Location:** Lines 35-37

```python
PUBLICATION_MODE = True  # High-quality publication graphs
DPI = 600  # 600 DPI for journals (300 for preview)
FONT_FAMILY = 'serif'  # 'serif' for academic, 'sans-serif' for modern
```

**Options:**

| Setting | Value | Use Case |
|---------|-------|----------|
| `PUBLICATION_MODE` | `True` | Journal submission, conference papers |
| | `False` | Quick previews, presentations |
| `DPI` | `600` | Print publication (IEEE, ACM journals) |
| | `300` | Online publication, most conferences |
| | `150` | Fast preview |
| `FONT_FAMILY` | `'serif'` | Academic papers (Times-like fonts) |
| | `'sans-serif'` | Modern tech reports, slides |

---

### 4. Benchmark Selection

**Location:** Lines 39-42

```python
# Automatic detection (most recent)
AUTO_DETECT_BENCHMARK = True

# OR manual specification
AUTO_DETECT_BENCHMARK = False
MANUAL_BENCHMARK_NAME = 'benchmark_150k_001'
```

---

## New Visualizations Explained

### 1. **Overhead Analysis** (NEW)
**File:** `overhead_analysis.png`

**What it shows:** Percentage of time wasted on process management vs actual computation.

**Why it matters:**
- High overhead = parallel implementation is inefficient
- Helps identify if parallelization is worth the cost
- Shows which algorithms minimize overhead

**How to interpret:**
- Lower bars = better (less wasted time)
- Overhead > 50% = parallelization is counterproductive
- Overhead < 10% = excellent parallel efficiency

---

### 2. **Load Imbalance Analysis** (NEW)
**File:** `load_imbalance_analysis.png`

**What it shows:** How evenly work is distributed across processors.

**Metrics:**
- **Max/Min Ratio:** Ratio of heaviest to lightest processor workload
  - 1.0 = perfect balance
  - 2.0 = worst processor does 2× the work of best

- **Coefficient of Variation (CV):** Statistical measure of variability
  - 0% = perfect balance
  - Higher % = more imbalance

**Why it matters:**
- Explains why some strategies have better speedup
- Shows which strategy (BL, CL, BWT, CWT) balances load best
- CWT should show lowest imbalance (best balance)

**How to interpret:**
- BL typically has highest imbalance (worst)
- CWT typically has lowest imbalance (best)
- Lower values = processors finish at similar times

---

### 3. **Strong Scaling Analysis** (NEW)
**File:** `strong_scaling_analysis.png`

**What it shows:** How performance scales with increasing processors.

**Two plots:**

**Plot 1 - Speedup vs Processors:**
- Shows actual speedup compared to ideal linear speedup
- Includes Amdahl's Law theoretical limits
- Shows diminishing returns as processors increase

**Plot 2 - Parallel Efficiency:**
- Shows processor utilization percentage
- Ideal = 100% (all processors fully utilized)
- Decreasing efficiency shows overhead impact

**Why it matters:**
- Helps determine optimal processor count
- Shows scalability limits
- Validates theoretical predictions (Amdahl's Law)

**How to interpret:**
- Speedup should increase linearly (follow ideal line)
- Gap between actual and ideal = overhead and imbalance
- Efficiency typically decreases as processors increase
- Super-linear speedup (>100% efficiency) = cache effects

---

### 4. **Cost-Benefit Analysis** (NEW)
**File:** `cost_benefit_analysis.png`

**What it shows:** Economic value of adding more processors.

**Two plots:**

**Plot 1 - Speedup per Processor:**
- Speedup divided by number of processors
- Shows "bang for buck" - benefit per resource unit
- Declining curve = diminishing returns

**Plot 2 - Marginal Speedup:**
- Additional speedup gained by adding one more processor
- Shows where adding processors stops being worthwhile

**Why it matters:**
- Helps make resource allocation decisions
- Shows point of diminishing returns
- Justifies infrastructure costs

**How to interpret:**
- Higher speedup/processor = better resource utilization
- Declining marginal speedup = diminishing returns
- When marginal speedup < 1.0, adding processors is inefficient

---

## Complete Example Configurations

### Example 1: Conference Paper - Compare 3 Processor Counts
```python
# Configuration for conference paper comparing scaling
PROCESSORS_TO_SHOW = [1, 4, 8]
SUPPORT_THRESHOLDS_TO_SHOW = [0.0008, 0.001, 0.002]
PUBLICATION_MODE = True
DPI = 300  # Conference standard
FONT_FAMILY = 'serif'
AUTO_DETECT_BENCHMARK = False
MANUAL_BENCHMARK_NAME = 'benchmark_150k_001'
```

**Result:** Clean comparison showing baseline (1), moderate (4), and high (8) parallelization across 3 support levels.

---

### Example 2: Journal Article - Full Analysis
```python
# Configuration for comprehensive journal paper
PROCESSORS_TO_SHOW = None  # Show all available
SUPPORT_THRESHOLDS_TO_SHOW = None  # Show all thresholds
PUBLICATION_MODE = True
DPI = 600  # High-quality print
FONT_FAMILY = 'serif'
AUTO_DETECT_BENCHMARK = True
```

**Result:** Complete analysis with all data points for thorough research paper.

---

### Example 3: Single Threshold Deep-Dive
```python
# Focus on single support threshold, compare all processors
PROCESSORS_TO_SHOW = [1, 2, 4, 8, 16]
SUPPORT_THRESHOLDS_TO_SHOW = [0.001]  # Single threshold
PUBLICATION_MODE = True
DPI = 600
FONT_FAMILY = 'serif'
AUTO_DETECT_BENCHMARK = False
MANUAL_BENCHMARK_NAME = 'benchmark_200k_002'
```

**Result:** Detailed processor scaling analysis at one support level.

---

### Example 4: Quick Preview (Fast Generation)
```python
# Fast generation for checking results
PROCESSORS_TO_SHOW = [1, 8]  # Just extremes
SUPPORT_THRESHOLDS_TO_SHOW = [0.001]
PUBLICATION_MODE = False  # Lower quality, faster
DPI = 150
FONT_FAMILY = 'sans-serif'
AUTO_DETECT_BENCHMARK = True
```

**Result:** Quick preview graphs generated in seconds.

---

## Output Files

### Standard Comparisons (Enhanced)
1. **execution_time_comparison.png**
   - Bar chart comparing all algorithms
   - Publication-quality styling
   - Better font sizes and colors

2. **speedup_comparison.png**
   - Line plot with clear markers
   - Baseline reference line
   - Value annotations

3. **efficiency_comparison.png**
   - Bar chart with 100% ideal line
   - Percentage labels
   - Auto-scaled y-axis

### Advanced Analytics (NEW)
4. **overhead_analysis.png**
   - Shows time wasted on parallelization overhead
   - Helps justify parallel implementation

5. **load_imbalance_analysis.png**
   - Statistical analysis of workload distribution
   - Explains performance differences between strategies

6. **strong_scaling_analysis.png**
   - Speedup and efficiency vs processor count
   - Includes Amdahl's Law theoretical limits
   - Shows scalability limits

7. **cost_benefit_analysis.png**
   - Economic analysis of parallelization
   - Marginal returns and optimal processor count
   - Resource allocation guidance

---

## Tips for Research Papers

### For Introduction/Background
- Use **execution_time_comparison.png** to show problem scale
- Use **speedup_comparison.png** to demonstrate improvement

### For Methodology Section
- Use **load_imbalance_analysis.png** to explain strategy differences
- Reference mathematical analysis document

### For Results Section
- Use **strong_scaling_analysis.png** for scalability discussion
- Use **efficiency_comparison.png** to show processor utilization
- Use **overhead_analysis.png** to discuss implementation efficiency

### For Discussion/Analysis
- Use **cost_benefit_analysis.png** to justify optimal configuration
- Compare against Amdahl's Law predictions from strong_scaling_analysis

### For Conclusion
- Summarize with **speedup_comparison.png** showing best strategy
- Use **cost_benefit_analysis.png** to recommend processor count

---

## Customization Beyond Configuration

### Change Color Scheme
Edit the `COLORS` dictionary (lines 70-79):
```python
COLORS = {
    'WDPA-CWT': '#FF5733',  # Custom red-orange
    # ... customize all colors
}
```

### Adjust Figure Sizes
For wider graphs (better for papers):
```python
fig, ax = plt.subplots(figsize=(14, 6))  # Wider
```

For taller graphs:
```python
fig, ax = plt.subplots(figsize=(10, 8))  # Taller
```

### Change Font Sizes
All font sizes are controlled via:
```python
plt.rcParams.update({
    'font.size': 11,        # Base font
    'axes.labelsize': 12,   # Axis labels
    'axes.titlesize': 13,   # Plot titles
    'legend.fontsize': 10,  # Legend
})
```

---

## Troubleshooting

### Issue: "No processor scaling detected"
**Cause:** Your benchmark didn't test multiple processor counts.

**Solution:**
- Check your benchmark config has `"num_processors": [1, 2, 4, 8]`
- Set `PROCESSORS_TO_SHOW = None` to see what's available

### Issue: Graphs are cut off
**Cause:** Figure size too small for content.

**Solution:** Increase DPI or figure size:
```python
DPI = 600
fig, ax = plt.subplots(figsize=(14, 7))
```

### Issue: Too much white space
**Cause:** `bbox_inches='tight'` adds padding.

**Solution:** Adjust padding:
```python
plt.savefig('plot.png', dpi=DPI, bbox_inches='tight', pad_inches=0.05)
```

### Issue: Filters not working
**Cause:** Data doesn't match filter values exactly.

**Solution:** Check available values first:
```python
PROCESSORS_TO_SHOW = None
SUPPORT_THRESHOLDS_TO_SHOW = None
```
Run once to see console output showing available values, then filter.

---

## Example Workflow

### Step 1: Run Benchmark
```bash
python run_benchmark.py
```

### Step 2: Preview Results
```python
# Set in visualize_results_enhanced.py
PUBLICATION_MODE = False
DPI = 150
PROCESSORS_TO_SHOW = None
SUPPORT_THRESHOLDS_TO_SHOW = None
```
```bash
python visualize_results_enhanced.py
```

### Step 3: Identify Key Results
Look at console output:
```
Processors: [1, 2, 4, 8, 16]
Support Thresholds: [0.0008, 0.001, 0.0015, 0.002, 0.003]
```

### Step 4: Filter for Paper
```python
# Configure for final publication
PUBLICATION_MODE = True
DPI = 600
PROCESSORS_TO_SHOW = [1, 4, 8]  # Based on step 3
SUPPORT_THRESHOLDS_TO_SHOW = [0.0008, 0.001, 0.002]
```

### Step 5: Generate Publication Plots
```bash
python visualize_results_enhanced.py
```

### Step 6: Use in Paper
- Plots saved in `results/{benchmark_name}/publication_plots/`
- 600 DPI suitable for print journals
- Serif fonts match academic style
- All graphs have consistent styling

---

## FAQ

**Q: Can I generate both filtered and unfiltered plots?**

A: Yes! Change output directory:
```python
output_dir = f'results/{benchmark_name}/filtered_plots'  # For filtered
output_dir = f'results/{benchmark_name}/all_plots'       # For unfiltered
```

**Q: How do I make graphs landscape vs portrait?**

A: Adjust figsize tuple:
```python
figsize=(12, 6)   # Landscape (width, height)
figsize=(8, 10)   # Portrait
figsize=(10, 10)  # Square
```

**Q: Can I export to other formats?**

A: Yes! Change save format:
```python
plt.savefig(f'{output_dir}/plot.pdf', dpi=DPI)  # PDF (vector)
plt.savefig(f'{output_dir}/plot.svg', dpi=DPI)  # SVG (vector)
plt.savefig(f'{output_dir}/plot.eps', dpi=DPI)  # EPS (LaTeX)
```

**Q: Which format for LaTeX papers?**

A: PDF or EPS work best with LaTeX. Add to your .tex:
```latex
\includegraphics[width=\textwidth]{plots/speedup_comparison.pdf}
```

---

## Summary

The enhanced visualization script provides:

✅ **Publication-Quality Styling** - Ready for journals and conferences
✅ **Configurable Filtering** - Show exactly what you need
✅ **4 New Advanced Analytics** - Deeper insights into performance
✅ **Consistent Styling** - All graphs match for professional papers
✅ **High DPI Output** - Print-ready quality
✅ **Flexible Configuration** - Easy customization without code changes

Perfect for your COS 781 research paper!
