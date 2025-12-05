import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
import multiprocessing
from typing import Optional, Dict

def geometric_hill(x, theta, alpha, gamma, x_marginal=None):
    x_decayed = np.zeros_like(x)
    for i in range(len(x)):
        x_decayed[i] = x[i] if i == 0 else x[i] + theta * x_decayed[i - 1]
    inflexion = np.dot(np.array([1 - gamma, gamma]), np.array([np.min(x_decayed), np.max(x_decayed)]))
    if x_marginal is None:
        x_scurve = x_decayed ** alpha / (x_decayed ** alpha + inflexion ** alpha)
    else:
        x_scurve = x_marginal ** alpha / (x_marginal ** alpha + inflexion ** alpha)
    return x_scurve

CHANNELS = [
    ('TV Spends', 0.3, 0.9, 0.8),
    ('Meta Video Spends', 0, 0.5, 0.8),
    ('Branded Paid Search Spends', 0.1, 2.6, 0.5),
    ('Outdoor Spends', 0.3, 0.9, 0.4),
    ('Programmatic Display Spends', 0.8, 0.9, 1),
    ('Radio Spends', 0.1, 3, 0.3),
    ('Direct Display Spends', 0.3, 3, 1),
    ('Youtube Spends', 0.3, 1.2, 0.3),
    ('Competitor 1 ATL Spends', 0.2, 2.8, 1.0),
    ('Competitor 2 ATL Spends', 0.1, 3, 1),
    ('Competitor 3 ATL Spends', 0.3, 3, 0.3),
]
SEASONALITY = ['December Seasonality', 'February Seasonality']
BETA_COLS = [
    'Beta_Base',
    'Beta_TV Spends',
    'Beta_Meta Video Spends',
    'Beta_Branded Paid Search Spends',
    'Beta_Outdoor Spends',
    'Beta_Programmatic Display Spends',
    'Beta_Radio Spends',
    'Beta_Direct Display Spends',
    'Beta_Youtube Spends',
    'Beta_Competitor 1 ATL Spends',
    'Beta_Competitor 2 ATL Spends',
    'Beta_Competitor 3 ATL Spends',
    'Beta_December Seasonality',
    'Beta_February Seasonality',
]


class ROIOptimizer:
    """
    ROI-based optimizer that pre-calculates achievable sales range on initialization.
    Uses the same optimization algorithm as DefaultOptimizer but with ROI ordering preference.
    """
    
    def __init__(self, csv_path='data/Model_data_for_simulator.csv', 
                 roi_csv_path="data/marketing_roas.csv",
                 algorithm='basinhopping',
                 de_maxiter=500, de_seed=42, de_polish=True, 
                 niter_bh=100, maxiter_sa=1000):
        """
        Initialize ROI optimizer and calculate achievable sales range.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file with model data
        roi_csv_path : str
            Path to the CSV file with ROI/ROAS data
        algorithm : str
            Global optimization algorithm: 'basinhopping', 'differential_evolution', 
            'dual_annealing', or 'multistart'
        """
        self.csv_path = csv_path
        self.roi_csv_path = roi_csv_path
        self.algorithm = algorithm
        self.de_maxiter = de_maxiter
        self.de_seed = de_seed
        self.de_polish = de_polish
        self.niter_bh = niter_bh
        self.maxiter_sa = maxiter_sa
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.jan24_idx = self.df[self.df['Month'] == 'Jan-24'].index[0]
        self.betas = self.df.loc[self.jan24_idx-1, BETA_COLS].values.astype(float)
        self.base = 1
        
        # Load ROI mapping
        roi_df = pd.read_csv(roi_csv_path)
        self.roi_map = dict(zip(roi_df['Marketing Inputs'], roi_df['ROAS']))
        
        # Rank channels by ROI (descending order)
        self.ranked_channels = sorted(
            [(ch, self.roi_map[ch]) for ch in self.roi_map if ch in [c[0] for c in CHANNELS]],
            key=lambda x: x[1],
            reverse=True
        )
        self.ranked_channel_names = [ch for ch, _ in self.ranked_channels]
        
        # Calculate max spends
        self.max_spends = {}
        for ch, *_ in CHANNELS:
            self.max_spends[ch] = self.df.loc[:self.jan24_idx-1, ch].astype(float).max()
        
        self.seasonality = [0, 0]  # Jan-24: neither December nor February
        self.bounds = [(0, 1.3 * self.max_spends[ch]) for ch, *_ in CHANNELS]
        
        # Pre-calculate achievable range
        self.max_achievable_sales = None
        self.min_achievable_sales = None
        self.max_achievable_spends = None
        self.min_achievable_spends = None
    
    def predict_sales(self, spends):
        """Predict sales for given spends."""
        adstocked = []
        for i, (ch, theta, alpha, gamma) in enumerate(CHANNELS):
            hist = self.df.loc[:self.jan24_idx-1, ch].astype(float).values
            hist = np.append(hist, spends[i])
            adstocked_val = geometric_hill(hist, theta, alpha, gamma)[-1]
            adstocked.append(adstocked_val)
        features = [self.base] + adstocked + list(self.seasonality)
        return float(np.dot(self.betas, features))
    
    def roi_penalty(self, spends):
        """Calculate penalty for violating ROI ordering (higher ROI should have higher spend)."""
        penalty = 0
        for i in range(len(self.ranked_channel_names) - 1):
            ch_high = self.ranked_channel_names[i]
            ch_low = self.ranked_channel_names[i+1]
            idx_high = [idx for idx, (c, *_) in enumerate(CHANNELS) if c == ch_high][0]
            idx_low = [idx for idx, (c, *_) in enumerate(CHANNELS) if c == ch_low][0]
            # Penalize if spend for higher ROI channel is less than lower ROI channel
            if spends[idx_high] < spends[idx_low]:
                penalty += (spends[idx_low] - spends[idx_high]) ** 2
        return penalty
    
    def initialize_sales_range(self):
        """
        Calculate maximum and minimum achievable sales.
        This should be called once during initialization.
        """
        print(f"Initializing sales range using {self.algorithm}...")
        
        # Find maximum achievable sales
        def maximize_sales(spends):
            sales = -self.predict_sales(spends)
            # Apply ROI penalty to prefer higher ROI channels
            roi_pen = self.roi_penalty(spends)
            return sales + 0.01 * roi_pen  # Small weight so sales still dominate
        
        print(f"Finding maximum achievable sales...")
        result_max = self._run_optimization(maximize_sales)
        
        if result_max and result_max.x is not None:
            self.max_achievable_sales = self.predict_sales(result_max.x)
            self.max_achievable_spends = result_max.x
            print(f"Maximum achievable sales: {self.max_achievable_sales:.2f}")
        else:
            print("Warning: Failed to find maximum achievable sales")
        
        # Find minimum achievable sales
        def minimize_sales(spends):
            sales = self.predict_sales(spends)
            roi_pen = self.roi_penalty(spends)
            return sales + 0.01 * roi_pen
        
        print(f"Finding minimum achievable sales...")
        result_min = self._run_optimization(minimize_sales)
        
        if result_min and result_min.x is not None:
            self.min_achievable_sales = self.predict_sales(result_min.x)
            self.min_achievable_spends = result_min.x
            print(f"Minimum achievable sales: {self.min_achievable_sales:.2f}")
        else:
            print("Warning: Failed to find minimum achievable sales")
        
        if self.max_achievable_sales and self.min_achievable_sales:
            print(f"Achievable sales range: [{self.min_achievable_sales:.2f}, {self.max_achievable_sales:.2f}]")
    
    def _run_optimization(self, objective):
        """Run optimization with the configured algorithm."""
        if self.algorithm == 'basinhopping':
            x0 = [0.5 * 1.3 * self.max_spends[ch] for ch, *_ in CHANNELS]
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": self.bounds}
            result = basinhopping(
                objective, x0=x0, niter=self.niter_bh,
                minimizer_kwargs=minimizer_kwargs,
                seed=self.de_seed, T=1.0
            )
        elif self.algorithm == 'differential_evolution':
            num_workers = min(multiprocessing.cpu_count(), 4)
            result = differential_evolution(
                objective, bounds=self.bounds, seed=self.de_seed,
                maxiter=self.de_maxiter, polish=self.de_polish,
                atol=1e-6, tol=0.01, workers=num_workers,
                popsize=10, updating='immediate'
            )
        elif self.algorithm == 'dual_annealing':
            result = dual_annealing(
                objective, bounds=self.bounds,
                maxiter=self.maxiter_sa, seed=self.de_seed
            )
        elif self.algorithm == 'multistart':
            best_result = None
            best_value = float('inf')
            n_starts = 20
            for i in range(n_starts):
                x0 = np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds])
                res = minimize(objective, x0, bounds=self.bounds, method='L-BFGS-B')
                if res.success and res.fun < best_value:
                    best_value = res.fun
                    best_result = res
            result = best_result
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return result
    
    def optimize(self, target_sales: float) -> Optional[Dict]:
        """
        Optimize spends for target sales with ROI ordering preference.
        
        Parameters:
        -----------
        target_sales : float
            Target sales value to achieve
            
        Returns:
        --------
        dict or None
            Dictionary with spends, seasonality, predicted_sales, and achievable range.
            Returns None if optimization fails.
        """
        if self.max_achievable_sales is None or self.min_achievable_sales is None:
            raise RuntimeError("Optimizer not initialized. Call initialize_sales_range() first.")
        
        # Check if target is achievable
        if target_sales > self.max_achievable_sales:
            print(f"Target {target_sales:.2f} exceeds maximum achievable ({self.max_achievable_sales:.2f})")
            return {
                'spends': {ch: self.max_achievable_spends[i] for i, (ch, *_) in enumerate(CHANNELS)},
                'seasonality': {s: self.seasonality[i] for i, s in enumerate(SEASONALITY)},
                'predicted_sales': self.max_achievable_sales,
                'min_achievable_sales': self.min_achievable_sales,
                'max_achievable_sales': self.max_achievable_sales,
                'target_achievable': False,
                'message': f'Target sales {target_sales:.2f} exceeds maximum achievable {self.max_achievable_sales:.2f}'
            }
        
        if target_sales < self.min_achievable_sales:
            print(f"Target {target_sales:.2f} is below minimum achievable ({self.min_achievable_sales:.2f})")
            return {
                'spends': {ch: self.min_achievable_spends[i] for i, (ch, *_) in enumerate(CHANNELS)},
                'seasonality': {s: self.seasonality[i] for i, s in enumerate(SEASONALITY)},
                'predicted_sales': self.min_achievable_sales,
                'min_achievable_sales': self.min_achievable_sales,
                'max_achievable_sales': self.max_achievable_sales,
                'target_achievable': False,
                'message': f'Target sales {target_sales:.2f} is below minimum achievable {self.min_achievable_sales:.2f}'
            }
        
        # Target is achievable, optimize to hit it with ROI preference
        print(f"Optimizing for target sales: {target_sales:.2f}")
        
        def objective(spends):
            predicted = self.predict_sales(spends)
            sales_error = (predicted - target_sales) ** 2
            roi_pen = self.roi_penalty(spends)
            return sales_error + 1e6 * roi_pen  # High weight for ROI ordering
        
        result = self._run_optimization(objective)
        
        # Refine with L-BFGS-B
        if result and result.x is not None:
            result = minimize(objective, result.x, bounds=self.bounds, method='L-BFGS-B')
        
        if not result or result.x is None:
            print("Optimization failed")
            return None
        
        spends = result.x
        predicted_sales = self.predict_sales(spends)
        
        return {
            'spends': {ch: spends[i] for i, (ch, *_) in enumerate(CHANNELS)},
            'seasonality': {s: self.seasonality[i] for i, s in enumerate(SEASONALITY)},
            'predicted_sales': predicted_sales,
            'min_achievable_sales': self.min_achievable_sales,
            'max_achievable_sales': self.max_achievable_sales,
            'target_achievable': True
        }
def roi_based_optimizer(target_sales, roi_csv_path="data/marketing_roas.csv", csv_path='data/Model_data_for_simulator.csv'):
    """
    Legacy function for backward compatibility.
    Creates a new optimizer instance each time (not efficient).
    For production use, create an ROIOptimizer instance and reuse it.
    """
    optimizer = ROIOptimizer(
        csv_path=csv_path,
        roi_csv_path=roi_csv_path,
        algorithm='basinhopping',
        niter_bh=50
    )
    optimizer.initialize_sales_range()
    return optimizer.optimize(target_sales)

def print_spends_and_sales(results, target_sales):
    """
    Print the spends, seasonality, predicted sales, and validation for the optimizer result.
    """
    if results is None:
        print("No results to display.")
        return

    spends_dict = results.get("spends", {})
    seasonality_dict = results.get("seasonality", {})
    predicted_sales = results.get("predicted_sales", None)
    min_achievable = results.get("min_achievable_sales", None)
    max_achievable = results.get("max_achievable_sales", None)

    print("\n--- Validation Solution ---")
    print("Target Sales:", target_sales)
    
    if min_achievable is not None and max_achievable is not None:
        print(f"Achievable Sales Range: [{min_achievable:.2f}, {max_achievable:.2f}]")
    
    print("Predicted Sales:", f"{predicted_sales:.2f}" if predicted_sales is not None else "N/A")
    print("\nSpends:")
    for ch, val in spends_dict.items():
        print(f"  {ch}: {val:.2f}")
    print("\nSeasonality:")
    for s, val in seasonality_dict.items():
        print(f"  {s}: {val}")

    # Optionally, print difference and check closeness
    if predicted_sales is not None:
        diff = abs(predicted_sales - target_sales)
        print(f"\nDifference: {diff:.2f}")
        if np.isclose(predicted_sales, target_sales, rtol=1e-3, atol=1e-2):
            print("Validation: SUCCESS (Predicted sales is close to target sales)")
        else:
            print("Validation: WARNING (Predicted sales is NOT close to target sales)")

def main():
    target_sales = 7000000
    
    print("=" * 60)
    print("Initializing ROI Optimizer")
    print("=" * 60)
    
    # Create optimizer instance and initialize (done once)
    optimizer = ROIOptimizer(algorithm='basinhopping', niter_bh=50)
    optimizer.initialize_sales_range()
    
    print("\n" + "=" * 60)
    print(f"Optimizing for target: {target_sales:,.0f}")
    print("=" * 60)
    
    # Optimize for target (can be called many times)
    results = optimizer.optimize(target_sales)
    print_spends_and_sales(results, target_sales)

if __name__ == "__main__":
    main()