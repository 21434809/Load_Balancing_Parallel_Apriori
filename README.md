# Load Balancing Parallel Apriori

A data mining project implementing parallel Apriori algorithm with load balancing for frequent itemset mining on hypermarket datasets.

## Project Overview

This project implements the Apriori algorithm with parallel processing capabilities and load balancing techniques to efficiently mine frequent itemsets from large transactional datasets, specifically using hypermarket transaction data.

## Features

- Parallel implementation of the Apriori algorithm
- Load balancing across multiple processes/threads
- Efficient handling of large datasets
- Support for configurable minimum support thresholds
- Visualization of frequent itemsets

## Dataset

The project uses the hypermarket dataset from the DSFSI datasets repository:
- Source: [DSFSI Datasets - Hypermarket](https://media.githubusercontent.com/media/dsfsi/dsfsi-datasets/refs/heads/master/data/cos781/hypermarket_dataset.csv)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Load_Balancing_Parallel_Apriori
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

## Project Structure

```
Load_Balancing_Parallel_Apriori/
├── main.py                 # Main execution script
├── src/
│   └── apriori.py         # Apriori algorithm implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── LICENSE              # License file
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Academic Context

This project is part of COS 781 Data Mining coursework, focusing on parallel algorithms and load balancing techniques in data mining applications.
