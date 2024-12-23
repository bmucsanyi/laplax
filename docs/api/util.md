# Utilities Module

The utilities module provides core functionality used throughout the library, from PyTree operations to matrix-vector products and data loading.

## Overview

### PyTree Operations
The tree utilities provide comprehensive functionality for:

- **Basic Operations**: Add, subtract, multiply, and other mathematical operations on PyTrees
- **Statistical Functions**: Mean, variance, covariance computations for PyTree structures
- **Array Creation**: Generate ones, zeros, random numbers in PyTree format
- **Matrix Operations**: Matrix-vector products with PyTree structures

### Matrix-Vector Products
Matrix-free operations for efficient computation:

- **Diagonal Extraction**: Compute diagonals without materializing full matrices
- **Dense Conversion**: Convert matrix-vector products to dense matrices when needed
- **Basis Operations**: Create and manipulate basis vectors for matrix operations

### Data Loading
Utilities for handling data in batches:

- **Batch Processing**: Functions for processing data loaders and iterables
- **Data Transformations**: Tools for splitting and transforming batch data
- **Reduction Operations**: Methods for reducing results across batches

### Adaptive Operations
Functions for flexible computation:

- **Batch Size Control**: Environment-variable controlled batch sizes
- **Data Type Management**: Configurable precision and data types
- **Precomputation**: Optional precomputation of repeated operations

### Flattening Operations
Tools for converting between PyTrees and flat arrays:

- **Flattening**: Convert PyTrees to flat arrays
- **Unflattening**: Reconstruct PyTrees from flat arrays
- **Function Wrapping**: Wrap functions to handle flattened inputs/outputs
