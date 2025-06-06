"""
EQTransformer Indonesia Training Module
Core Indonesia (core_ind) subdirectory for Indonesian seismic data training

This module contains specialized training scripts for Indonesian seismic data
with extended input dimensions (30085 samples, 300.8 seconds @ 100Hz)
"""

from .train_indonesia_model import (
    check_data_compatibility,
    train_indonesia_eqt_full,
    train_indonesia_eqt_separate,
    create_combined_dataset,
    train_combined_dataset
)

__all__ = [
    'check_data_compatibility',
    'train_indonesia_eqt_full', 
    'train_indonesia_eqt_separate',
    'create_combined_dataset',
    'train_combined_dataset'
] 