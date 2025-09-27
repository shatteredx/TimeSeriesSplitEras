# TimeSeriesSplitEras

A robust time series cross-validator for financial prediction competitions with era-based data and embargo periods. Built on top of scikit-learn's `_BaseKFold`, this tool is specifically designed for competitions like Numerai where temporal data leakage prevention is critical.

## Features

- **Era-based splitting**: Handles data organized by discrete time periods (eras) rather than continuous timestamps
- **Embargo periods**: Automatically excludes a configurable number of eras between training and test sets to prevent data leakage
- **Expanding window**: Implements a walk-forward validation approach where training data expands over time
- **Flexible era formats**: Supports integer eras (1, 2, 3) and string eras ("1", "0001", "era_001")
- **Sklearn compatible**: Follows scikit-learn's cross-validation API conventions
- **Debug mode**: Detailed logging of split information for validation and troubleshooting

## Why TimeSeriesSplitEras?

Standard `TimeSeriesSplit` from scikit-learn doesn't account for:
1. **Era-based data structure** - Financial data is often grouped into discrete eras rather than continuous time
2. **Embargo periods** - In real-world trading, there's a delay between observation and action
3. **Flexible era identifiers** - Competitions may use various era naming conventions

This implementation solves these problems while maintaining compatibility with scikit-learn's ecosystem.

## Installation

Simply copy the `time_series_split_eras.py` file to your project:


**Requirements:**
```
scikit-learn>=0.24.0
numpy>=1.19.0
pandas>=1.1.0
```

## Quick Start

```python
import pandas as pd
from time_series_split_eras import TimeSeriesSplitEras

# Load your data with an era column
df = pd.read_csv('numerai_training_data.csv')

# Initialize the splitter
tscv = TimeSeriesSplitEras(
    n_splits=5,
    embargo_size=4,
    min_train_ratio=0.5,
    era_col='era',
    debug=True
)

# Use in cross-validation
for train_idx, test_idx in tscv.split(df):
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    # Your model training and evaluation here
    # model.fit(train_data[features], train_data[target])
    # predictions = model.predict(test_data[features])
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | int | 5 | Number of cross-validation folds |
| `embargo_size` | int | 30 | Number of eras to exclude between train and test sets |
| `min_train_ratio` | float | 0.5 | Ratio of total eras for initial training (0 to 1) |
| `era_col` | str | 'era' | Name of the column containing era identifiers |
| `debug` | bool | False | Print detailed split information |

## Detailed Usage

### Basic Example with Numerai Data

```python
from time_series_split_eras import TimeSeriesSplitEras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load Numerai dataset
train_df = pd.read_csv('numerai_training_data.csv')
feature_cols = [f for f in train_df.columns if f.startswith('feature_')]

# Initialize cross-validator
cv = TimeSeriesSplitEras(
    n_splits=5,
    embargo_size=4,        # Embargo 4 eras between train/test
    min_train_ratio=0.6,   # Use 60% of data for initial training
    era_col='era',
    debug=True
)

# Cross-validation loop
scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(train_df)):
    # Split data
    X_train = train_df.iloc[train_idx][feature_cols]
    y_train = train_df.iloc[train_idx]['target']
    X_test = train_df.iloc[test_idx][feature_cols]
    y_test = train_df.iloc[test_idx]['target']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    scores.append(score)
    print(f"Fold {fold + 1} Score: {score:.4f}")

print(f"\nMean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

### Using with Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Cross-validate pipeline
for train_idx, test_idx in tscv.split(df):
    X_train, X_test = df.iloc[train_idx][features], df.iloc[test_idx][features]
    y_train, y_test = df.iloc[train_idx]['target'], df.iloc[test_idx]['target']
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Score: {score:.4f}")
```

### Handling Different Era Formats

```python
# Integer eras: 1, 2, 3, ..., 1000
df_int = pd.DataFrame({'era': [1, 1, 2, 2, 3, 3], 'value': [1, 2, 3, 4, 5, 6]})

# String eras without padding: "1", "2", "3", ..., "1000"  
df_str = pd.DataFrame({'era': ['1', '1', '2', '2', '10', '10'], 'value': [1, 2, 3, 4, 5, 6]})

# String eras with padding: "0001", "0002", ..., "1000"
df_padded = pd.DataFrame({'era': ['0001', '0001', '0002', '0002'], 'value': [1, 2, 3, 4]})

# All work correctly with numerical sorting
tscv = TimeSeriesSplitEras(n_splits=2, embargo_size=1, min_train_ratio=0.5)
for train_idx, test_idx in tscv.split(df_str):
    print(f"Train: {df_str.iloc[train_idx]['era'].unique()}")
    print(f"Test: {df_str.iloc[test_idx]['era'].unique()}")
```

## How It Works

### Split Strategy

1. **Initial Training Window**: Determined by `min_train_ratio` × total eras
2. **Embargo Period**: Last `embargo_size` eras of training are excluded
3. **Test Windows**: Remaining eras are divided into `n_splits` equal test periods
4. **Expanding Window**: Training data expands with each split, test window moves forward

### Visual Example

```
Total eras: 100
min_train_ratio: 0.6 (60 eras initial)
embargo_size: 4
n_splits: 5

Split 1: Train [1-56]  | Embargo [57-60] | Test [61-68]
Split 2: Train [1-64]  | Embargo [65-68] | Test [69-76]
Split 3: Train [1-72]  | Embargo [73-76] | Test [77-84]
Split 4: Train [1-80]  | Embargo [81-84] | Test [85-92]
Split 5: Train [1-88]  | Embargo [89-92] | Test [93-100]
```

## Debug Output

When `debug=True`, you'll see detailed information:

```
Total number of eras: 120
Era type: int
First era: 1
Last era: 120

Split 1/5
Train eras: 1 to 56 (total: 56)
Embargo eras: 57 to 60 (total: 4)
Test eras: 61 to 72 (total: 12)

Split 2/5
Train eras: 1 to 68 (total: 68)
Embargo eras: 69 to 72 (total: 4)
Test eras: 73 to 84 (total: 12)
...
```

## Error Handling

The validator will raise helpful errors for common issues:

```python
# Not enough eras for configuration
ValueError: Not enough eras (50) for the configuration. 
After initial training (40), only 10 eras remain for 5 test splits

# Invalid train ratio
ValueError: min_train_ratio must be between 0 and 1

# Missing era column
ValueError: era column 'era' not found in DataFrame

# Training period too small
ValueError: Initial training period (25 eras) must be larger than embargo size (30 eras)
```

## Comparison with Sklearn TimeSeriesSplit

| Feature | TimeSeriesSplit | TimeSeriesSplitEras |
|---------|-----------------|---------------------|
| Era-based splitting | ❌ | ✅ |
| Embargo periods | ❌ | ✅ |
| Flexible era formats | ❌ | ✅ |
| Expanding window | ✅ | ✅ |
| DataFrame support | ❌ | ✅ |
| Debug mode | ❌ | ✅ |

## Use Cases

Perfect for:
- **Numerai competition** cross-validation
- **Financial time series** with discrete periods
- **Trading strategy backtesting** with embargo requirements
- **Any temporal data** where leakage prevention is critical
- **Competition datasets** with era-based structure

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Testing

```python
# Run basic tests
python -m pytest tests/

# Or run the example script
python examples/numerai_example.py
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built on scikit-learn's cross-validation framework
- Designed for Numerai and similar financial prediction competitions
- Inspired by the need for proper temporal validation in quantitative finance

## Citation

If you use this in your research or competition submissions, please cite:

```bibtex
@software{timeseriesspliters,
  title = {TimeSeriesSplitEras: Era-based Time Series Cross-Validation with Embargo},
  author = {ShatteredX},
  year = {2025},
  url = {https://github.com/yourusername/TimeSeriesSplitEras}
}
```

## Questions or Issues?

- Open an issue on GitHub
- Check existing issues for solutions
- Read the documentation carefully

---

**Happy modeling! 📈**
