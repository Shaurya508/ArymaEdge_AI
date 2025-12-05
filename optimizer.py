import pandas as pd

def get_spend_limits(csv_path: str) -> dict:
    """Get historical max and mean for each spend channel."""
    df = pd.read_csv(csv_path)
    limits = {}
    for ch in df.columns:
        if "Spends" in ch:
            limits[ch] = {
                "max": float(df[ch].astype(float).max()),
                "mean": float(df[ch].astype(float).mean())
            }
    return limits


print(get_spend_limits("data\\Model_data_for_simulator.csv"))