"""
Statistical validation utilities for weather data

Implements hypothesis testing and uncertainty quantification
for grid monitoring applications.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


def kolmogorov_smirnov_test(data: np.ndarray, reference_dist: str = 'norm') -> Dict:
    """
    Test if data follows expected distribution using K-S test
    
    Returns:
        Dict with test statistic, p-value, and interpretation
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 10:
        return {'statistic': np.nan, 'p_value': np.nan, 'normal': False}
    
    if reference_dist == 'norm':
        # Standardize data
        standardized = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        statistic, p_value = stats.kstest(standardized, 'norm')
    else:
        statistic, p_value = stats.kstest(clean_data, reference_dist)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'normal': p_value > 0.05,  # Null hypothesis: data is normal
        'interpretation': 'normal' if p_value > 0.05 else 'non-normal'
    }


def detect_changepoints(timeseries: np.ndarray, window_size: int = 50) -> List[int]:
    """
    Detect structural breaks in time series using cumulative sum
    
    Returns:
        List of changepoint indices
    """
    if len(timeseries) < window_size * 2:
        return []
    
    # Calculate cumulative sum of deviations from mean
    mean_val = np.nanmean(timeseries)
    cusum = np.nancumsum(timeseries - mean_val)
    
    # Sliding window variance
    changepoints = []
    for i in range(window_size, len(cusum) - window_size):
        before = cusum[i-window_size:i]
        after = cusum[i:i+window_size]
        
        # Test for significant difference in means
        if len(before) > 5 and len(after) > 5:
            _, p_val = stats.ttest_ind(before, after)
            if p_val < 0.01:  # Significant change
                changepoints.append(i)
    
    return changepoints


def calculate_uncertainty_bounds(
    data: np.ndarray, 
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate uncertainty bounds using bootstrap confidence intervals
    
    Returns:
        Dict with mean, lower_bound, upper_bound, std_error
    """
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 10:
        return {
            'mean': np.nan,
            'lower_bound': np.nan, 
            'upper_bound': np.nan,
            'std_error': np.nan
        }
    
    # Bootstrap resampling
    n_bootstrap = 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'mean': np.mean(clean_data),
        'lower_bound': np.percentile(bootstrap_means, lower_percentile),
        'upper_bound': np.percentile(bootstrap_means, upper_percentile),
        'std_error': np.std(bootstrap_means)
    }


def validate_spatial_correlation(
    dataset: xr.Dataset,
    variables: List[str],
    max_distance_km: float = 500
) -> Dict:
    """
    Validate spatial correlation structure in gridded data
    
    Returns:
        Dict with correlation statistics and spatial autocorrelation metrics
    """
    results = {}
    
    for var in variables:
        if var not in dataset:
            continue
            
        # Get latest time slice to avoid memory issues
        data_slice = dataset[var].isel(time=-1)
        
        # Convert to DataFrame for correlation analysis
        df = data_slice.to_dataframe().reset_index()
        df = df.dropna()
        
        if len(df) < 100:
            results[var] = {'error': 'Insufficient valid data points'}
            continue
        
        # Calculate spatial lag correlation (simplified)
        # Sample subset of points to avoid computational explosion
        if len(df) > 1000:
            df_sample = df.sample(n=1000, random_state=42)
        else:
            df_sample = df
        
        # Calculate pairwise distances (simplified - assumes regular grid)
        coords = df_sample[['latitude', 'longitude']].values
        values = df_sample[var].values
        
        # Simple spatial autocorrelation using nearby points
        autocorr_values = []
        for i in range(min(100, len(coords))):
            # Find nearby points (simplified distance)
            distances = np.sqrt(
                ((coords - coords[i]) ** 2).sum(axis=1)
            )
            nearby_mask = (distances > 0) & (distances < 5)  # degrees
            
            if np.sum(nearby_mask) > 3:
                nearby_values = values[nearby_mask]
                correlation = np.corrcoef(values[i], nearby_values.mean())[0, 1]
                if not np.isnan(correlation):
                    autocorr_values.append(correlation)
        
        results[var] = {
            'spatial_autocorr_mean': np.mean(autocorr_values) if autocorr_values else np.nan,
            'spatial_autocorr_std': np.std(autocorr_values) if autocorr_values else np.nan,
            'n_points_analyzed': len(df_sample)
        }
    
    return results


def test_stationarity(timeseries: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test time series stationarity using Augmented Dickey-Fuller test
    
    Returns:
        Dict with test results and stationarity assessment
    """
    from statsmodels.tsa.stattools import adfuller
    
    # Remove NaN values
    clean_ts = timeseries[~np.isnan(timeseries)]
    
    if len(clean_ts) < 20:
        return {
            'stationary': False,
            'p_value': np.nan,
            'critical_values': {},
            'interpretation': 'insufficient_data'
        }
    
    try:
        result = adfuller(clean_ts)
        
        return {
            'stationary': result[1] <= alpha,
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'interpretation': 'stationary' if result[1] <= alpha else 'non_stationary'
        }
    except Exception as e:
        return {
            'stationary': False,
            'p_value': np.nan,
            'critical_values': {},
            'interpretation': f'error: {str(e)}'
        }


def validate_feature_importance(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: List[str],
    method: str = 'mutual_info'
) -> Dict[str, float]:
    """
    Calculate and validate feature importance using statistical methods
    
    Returns:
        Dict mapping feature names to importance scores
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
    
    # Remove rows with NaN values
    mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
    clean_features = features[mask]
    clean_target = target[mask]
    
    if len(clean_features) < 50:
        return {name: np.nan for name in feature_names}
    
    if method == 'mutual_info':
        scores = mutual_info_regression(clean_features, clean_target)
    elif method == 'random_forest':
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(clean_features, clean_target)
        scores = rf.feature_importances_
    else:
        # Correlation-based importance
        scores = []
        for i in range(clean_features.shape[1]):
            corr = np.corrcoef(clean_features[:, i], clean_target)[0, 1]
            scores.append(abs(corr) if not np.isnan(corr) else 0)
        scores = np.array(scores)
    
    return dict(zip(feature_names, scores))
