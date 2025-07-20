# Enhanced Weather Dataloader

Statistical validation and performance optimization for cloud-native PyTorch dataloaders using Zarr, Dask, and Xarray.

## Overview

Builds on [earth-mover/dataloader-demo](https://github.com/earth-mover/dataloader-demo) with added:
- Statistical data validation
- Performance benchmarking 
- Geospatial filtering
- Uncertainty quantification

## Features

- **Data Quality Checks**: Statistical validation of weather data integrity
- **Performance Metrics**: Throughput benchmarking with different configurations
- **Spatial Filtering**: Geographic bounding box and grid topology filtering
- **Error Handling**: Robust handling of missing data and network issues

## Usage

```python
from enhanced_dataloader import EnhancedWeatherDataset, validate_data

# Create dataset with validation
dataset = EnhancedWeatherDataset(
    source="arraylake",
    validate=True,
    bbox=(-125, 32, -114, 42)  # California region
)

# Benchmark performance
metrics = dataset.benchmark_performance(
    batch_sizes=[4, 8, 16],
    num_workers=[0, 4, 8]
)
```

## Files

- `enhanced_dataloader.py` - Main enhanced dataloader class
- `validation.py` - Statistical validation functions
- `benchmarks.py` - Performance testing utilities
- `spatial_utils.py` - Geospatial filtering functions
- `demo.ipynb` - Usage examples and benchmarks