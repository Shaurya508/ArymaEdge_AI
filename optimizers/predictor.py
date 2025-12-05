"""
Utility module for predicting sales from spend values.
"""
import numpy as np
import pandas as pd

# Import shared constants and functions from default_optimizer
from optimizers.default_optimizer import (
    geometric_hill,
    CHANNELS,
    BETA_COLS,
    SEASONALITY
)


def _get_seasonality_flags(month_str: str):
    """
    Seasonality rule:
    - If prediction month is December  => [Dec=1, Feb=0]
    - If prediction month is February => [Dec=0, Feb=1]
    - All other months                => [0, 0]
    """
    if not month_str:
        return (0, 0)

    month_part = str(month_str).split('-')[0].strip().lower()
    if month_part.startswith('dec'):
        return (1, 0)
    if month_part.startswith('feb'):
        return (0, 1)
    return (0, 0)


def predict_sales_for_spends(spends_dict, prediction_month="Jan-24", csv_path="data/Model_data_for_simulator.csv"):
    """
    Predict sales for given spend values without optimization.
    
    Args:
        spends_dict: Dictionary mapping channel names to spend values
        prediction_month: Month identifier (e.g., "Jan-24")
        csv_path: Path to the model data CSV
    
    Returns:
        float: Predicted sales value
    """
    df = pd.read_csv(csv_path)
    month_idx = df[df['Month'] == prediction_month].index[0]
    betas = df.loc[month_idx-1, BETA_COLS].values.astype(float)
    base = 1  # Base feature always 1 in dataset
    
    seasonality = _get_seasonality_flags(prediction_month)
    
    # Convert spends_dict to list in CHANNELS order
    spends_list = [spends_dict.get(ch[0], 0.0) for ch in CHANNELS]
    
    # Calculate adstocked values
    adstocked = []
    for i, (ch, theta, alpha, gamma) in enumerate(CHANNELS):
        hist = df.loc[:month_idx-1, ch].astype(float).values
        hist = np.append(hist, spends_list[i])
        adstocked_val = geometric_hill(hist, theta, alpha, gamma)[-1]
        adstocked.append(adstocked_val)
    
    # Calculate predicted sales
    features = [base] + adstocked + list(seasonality)
    predicted_sales = float(np.dot(betas, features))
    
    return predicted_sales

