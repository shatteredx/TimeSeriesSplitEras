from sklearn.model_selection._split import _BaseKFold
import numpy as np
import pandas as pd

class TimeSeriesSplitEras(_BaseKFold):
    """
    Time Series cross-validator with era-based embargo periods.
    
    Provides train/test indices to split time series data samples organized
    by eras (discrete time periods). Implements an expanding window approach
    where training data grows over time, with an embargo period between
    training and test sets to prevent data leakage.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/folds for cross-validation.
        
    embargo_size : int, default=30
        Number of eras to exclude between training and test sets to prevent
        data leakage from overlapping time periods.
        
    min_train_ratio : float, default=0.5
        Ratio of total eras to use for initial training period (including embargo).
        Must be between 0 and 1. For example, 0.5 means 50% of eras are used for
        initial training.
        
    era_col : str, default='era'
        Name of the column in the DataFrame containing era identifiers.
        Can be integers (1, 2, 3) or strings ("0001", "0002" or "1", "2", "3").
        
    debug : bool, default=False
        If True, prints detailed information about each split.
        
    Notes
    -----
    - The 'groups' parameter in split() is included for scikit-learn compatibility
      but is not used (era grouping is determined by the era_col).
    - Eras are automatically sorted numerically if they can be converted to integers,
      otherwise sorted lexicographically.
    """
    
    def __init__(self, n_splits=5, embargo_size=30, min_train_ratio=0.5, 
                 era_col='era', debug=False):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.embargo_size = embargo_size
        self.min_train_ratio = min_train_ratio
        self.era_col = era_col
        self.debug = debug
        
        if min_train_ratio is None or min_train_ratio <= 0 or min_train_ratio > 1:
            raise ValueError("min_train_ratio must be between 0 and 1")
    
    def split(self, X, y=None, feature_cols=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data containing the era column.
            
        y : array-like, optional
            Target variable (not used, included for sklearn compatibility).
            
        feature_cols : list of str, optional
            List of feature column names to include. If provided, only these
            columns plus the era column will be retained.
            
        groups : array-like, optional
            Group labels (not used, included for sklearn compatibility).
            Era grouping is determined by the era_col.
            
        Yields
        ------
        train_indices : np.ndarray
            The training set indices for that split.
            
        test_indices : np.ndarray
            The testing set indices for that split.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame containing the era column")
            
        if self.era_col not in X.columns:
            raise ValueError(f"era column '{self.era_col}' not found in DataFrame")
        
        # Validate feature columns if provided
        if feature_cols is not None:
            missing_cols = [col for col in feature_cols if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Feature columns {missing_cols} not found in DataFrame")
            required_cols = [self.era_col] + feature_cols
            X = X[required_cols]
            
        # Get unique eras and sort them appropriately
        unique_eras = X[self.era_col].unique()
        
        # Try to sort numerically (handles both int and string eras like "1", "2", "1000")
        try:
            unique_eras = sorted(unique_eras, key=lambda x: int(x))
        except (ValueError, TypeError):
            # If conversion fails, sort lexicographically (for eras like "era_001")
            unique_eras = sorted(unique_eras)
        
        n_eras = len(unique_eras)
        
        if self.debug:
            print(f"\nTotal number of eras: {n_eras}")
            print(f"Era type: {type(unique_eras[0]).__name__}")
            print(f"First era: {unique_eras[0]}")
            print(f"Last era: {unique_eras[-1]}")
            
        # Create era to index mapping
        era_to_idx = {era: X[X[self.era_col] == era].index.values for era in unique_eras}
        
        # Calculate initial training size based on min_train_ratio
        initial_train_size = int(np.ceil(n_eras * self.min_train_ratio))
        
        # Ensure we have enough eras for training + embargo
        if initial_train_size <= self.embargo_size:
            raise ValueError(
                f"Initial training period ({initial_train_size} eras) must be larger than "
                f"embargo size ({self.embargo_size} eras)"
            )
        
        # Calculate test size
        remaining_eras = n_eras - initial_train_size
        test_size_eras = remaining_eras // self.n_splits
        
        if test_size_eras <= 0:
            raise ValueError(
                f"Not enough eras ({n_eras}) for the configuration. "
                f"After initial training ({initial_train_size}), only {remaining_eras} eras "
                f"remain for {self.n_splits} test splits"
            )
        
        for i in range(self.n_splits):
            # Calculate test period
            test_start = initial_train_size + (i * test_size_eras)
            test_end = test_start + test_size_eras
            
            if test_end > n_eras:
                if self.debug:
                    print(f"Skipping split {i+1} as it would exceed available eras")
                continue
            
            # Training expands up to embargo period before test
            train_start = 0
            train_end = test_start - self.embargo_size
            
            # Embargo period boundaries (for debugging)
            embargo_start = train_end
            embargo_end = test_start
            
            # Get eras for each period
            train_eras = unique_eras[train_start:train_end]
            test_eras = unique_eras[test_start:test_end]
            
            if self.debug:
                print(f"\nSplit {i+1}/{self.n_splits}")
                print(f"Train eras: {train_eras[0]} to {train_eras[-1]} (total: {len(train_eras)})")
                print(f"Embargo eras: {unique_eras[embargo_start]} to {unique_eras[embargo_end-1]} "
                      f"(total: {embargo_end - embargo_start})")
                print(f"Test eras: {test_eras[0]} to {test_eras[-1]} (total: {len(test_eras)})")
            
            # Convert eras to indices
            train_indices = np.concatenate([era_to_idx[era] for era in train_eras])
            test_indices = np.concatenate([era_to_idx[era] for era in test_eras])
            
            yield train_indices, test_indices
