"""Validate manually entered model features."""
import numpy as np
import pandas as pd
from src.train_model import FEATURES, NUMERIC, CATEGORICAL

def validate_specs(specs, reference):
    if set(FEATURES)-set(specs):
        raise ValueError('All model specifications are required.')
    frame=pd.DataFrame([specs])[FEATURES]
    for col in NUMERIC:
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
        value=frame[col].iloc[0]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{col} must be finite and nonnegative.')
    for col in ['RAM_GB','Screen_Inches','Weight_KG','Total_Storage_GB']:
        if frame[col].iloc[0] <= 0:
            raise ValueError(f'{col} must be positive.')
    if frame.SSD_GB.iloc[0]+frame.HDD_GB.iloc[0] > frame.Total_Storage_GB.iloc[0]:
        raise ValueError('SSD plus HDD cannot exceed total storage.')
    for col in CATEGORICAL:
        if specs[col] not in reference[col].dropna().unique():
            raise ValueError(f'Unsupported {col}.')
    warnings=[f'{col} is outside the training range.' for col in NUMERIC if not reference[col].min() <= frame[col].iloc[0] <= reference[col].max()]
    return frame,warnings
