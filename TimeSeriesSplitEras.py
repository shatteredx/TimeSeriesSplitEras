from sklearn.model_selection._split import _BaseKFold
import numpy as np
import pandas as pd

class TimeSeriesSplitEras(_BaseKFold):
    """
    Time Series cross-validator with era-based embargo periods.
    Provides train/test indices to split time series data samples 
    that are observed at fixed time intervals.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splitting iterations.
    
    test_size : int, default=None
        Used to limit the size of the test set. If None,
        test_size will default to the size of the training set.
        
    embargo_size : int, default=30
        Number of eras to exclude between train and test sets.
        
    min_train_size : int, default=None
        Minimum size of the training set. If None, the first
        split will start with test_size + embargo_size samples.
    """
    
    def __init__(self, n_splits=5, test_size=None, 
                 embargo_size=30, min_train_size=None, era_col='era', debug=False):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_size = test_size
        self.embargo_size = embargo_size
        self.min_train_size = min_train_size
        self.era_col = era_col
        self.debug = debug
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
            
        era_col : str, default='era'
            Name of the column containing era information.
            
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
            
        Yields
        ------
        train : ndarray
            The training set indices for that split.
            
        test : ndarray
            The testing set indices for that split.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame containing the era column")
            
        if self.era_col not in X.columns:
            raise ValueError(f"era column '{self.era_col}' not found in DataFrame")
            
        # Get unique eras and ensure they're sorted
        unique_eras = sorted(X[self.era_col].unique())
        n_eras = len(unique_eras)
        
        # Validate parameters
        if self.test_size is None:
            test_size_eras = n_eras // (self.n_splits + 1)
        else:
            test_size_eras = self.test_size
            
        if self.min_train_size is None:
            min_train_size_eras = test_size_eras + self.embargo_size
        else:
            min_train_size_eras = self.min_train_size
            
        if n_eras < min_train_size_eras + test_size_eras + self.embargo_size:
            raise ValueError(
                f"Too few eras ({n_eras}) for the specified parameters: "
                f"min_train_size={min_train_size_eras}, "
                f"test_size={test_size_eras}, "
                f"embargo_size={self.embargo_size}"
            )
            
        # Create era to index mapping
        era_to_idx = {era: X[X[self.era_col] == era].index for era in unique_eras}
        
        # Generate the splits
        for i in range(self.n_splits):
            # Calculate test start era index (moving forward instead of backward)
            if self.test_size is None:
                test_size_eras = n_eras // (self.n_splits + 1)
            else:
                test_size_eras = self.test_size

            # Start from the earliest possible test period and move forward
            test_start = test_size_eras * i
            test_end = test_start + test_size_eras
            
            # Calculate train end era index (accounting for embargo)
            train_end = test_start - self.embargo_size
            
            # Always start from the first era for expanding window
            train_start = 0
            
            # Verify we have enough training data
            if self.min_train_size is not None and (train_end - train_start) < self.min_train_size:
                raise ValueError(f"Not enough training eras. Got {train_end - train_start}, need at least {self.min_train_size}")
            
            # Get eras for this split
            train_eras = unique_eras[train_start:train_end]
            test_eras = unique_eras[test_start:test_end]
            
            # Convert eras to indices
            train_indices = np.concatenate([era_to_idx[era] for era in train_eras])
            test_indices = np.concatenate([era_to_idx[era] for era in test_eras])
            
            # Print debug information if enabled
            if self.debug:
                print(f"\nSplit {i+1}/{self.n_splits}")
                print(f"Train eras: {train_eras[0]} to {train_eras[-1]} (total: {len(train_eras)})")
                print(f"Test eras: {test_eras[0]} to {test_eras[-1]} (total: {len(test_eras)})")
                print(f"Embargo eras: {unique_eras[train_end]} to {unique_eras[test_start-1]}")
                
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


def custom_cross_val_score(estimator, X, y, cv, scoring_func, era_wise=False, debug=False, **kwargs):
    """
    Evaluate a score by cross-validation using a custom scoring function.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
        
    X : array-like of shape (n_samples, n_features)
        Training data.
        
    y : array-like of shape (n_samples,)
        Target variable.
        
    cv : cross-validation generator
        Cross-validation splitter.
        
    scoring_func : callable
        Custom scoring function that takes y_true and y_pred as inputs
        and returns a score.
        
    era_wise : bool, default=False
        If True, passes the test DataFrame to the scoring function for era-wise calculations.

    debug : bool, default=False
        If True, prints additional information about each split and score.
        
    **kwargs : dict
        Additional parameters to be passed to the scoring function.
        
    Returns
    -------
    scores : array of float, shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Split the data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit and predict
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        # Calculate score
        if era_wise:
            score = scoring_func(y_test, y_pred, df=X_test, **kwargs)
        else:
            score = scoring_func(y_test, y_pred, **kwargs)
            
        if debug:
            print(f"\nSplit {i+1} Score: {score:.4f}")
            
        scores.append(score)
        
    return np.array(scores)


def era_wise_correlation(y_true, y_pred, df, era_col, prediction_name='prediction', target_col='target'):
    """
    Calculate era-wise correlation between predictions and targets.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    df : pandas.DataFrame
        DataFrame containing the era column
    era_col : str
        Name of the era column
    prediction_name : str, default='prediction'
        Name to use for prediction column
    target_col : str, default='target'
        Name of the target column
        
    Returns
    -------
    float
        Mean correlation across eras
    """
    # Create DataFrame with predictions and true values
    temp_df = pd.DataFrame({
        prediction_name: y_pred,
        target_col: y_true,
        era_col: df[era_col].values
    })
    
    # Calculate correlation for each era
    corrs = []
    for _, era_df in temp_df.groupby(era_col):
        corr_i = numerai_corr(era_df[[prediction_name]], era_df[target_col])
        corrs.append(corr_i[0])
    
    # Return mean correlation
    return np.mean(corrs)


# Example usage:
"""
# Initialize the custom splitter
cv = TimeSeriesSplitEras(
    n_splits=5,
    test_size=12,  # number of eras for test set
    embargo_size=30,  # number of eras to exclude
    min_train_size=100  # minimum number of eras for training
)

# Define a custom scoring function
def custom_score(y_true, y_pred, **kwargs):
    # Your custom scoring logic here
    return some_score

# Use the custom cross-validation
scores = custom_cross_val_score(
    estimator=your_model,
    X=your_data,
    y=your_target,
    cv=cv,
    scoring_func=custom_score,
    additional_param=value  # optional parameters for scoring function
)

# Print results
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
"""
