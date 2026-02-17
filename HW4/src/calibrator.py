# calibrator.py
from scipy.optimize import minimize_scalar
from sofr_swap import SOFRSwap
from yield_curve import YieldCurve
from curve_builder import SOFRCurveBuilder
import pandas as pd
from sofr_swap import SOFRSwap

class VolatilityCalibrator:
    """Calibrate volatility parameter to match market swap rates."""
    
    def __init__(self, curve_date, futures_filepath, market_swaps):
        """
        Parameters:
        -----------
        curve_date : str
            Valuation date
        futures_filepath : str
            Path to SOFR futures data
        market_swaps : dict
            {maturity: rate} e.g., {'2Y': 3.31095, '3Y': 3.32109}
        """
        self.curve_date = pd.to_datetime(curve_date).date()
        self.futures_filepath = futures_filepath
        self.market_swaps = market_swaps
        self.optimal_sigma = None
        self.optimal_curve = None
    
    def calculate_swap_errors(self, sigma):
        """Calculate swap pricing errors for a given sigma."""
        # Build curve with this sigma
        builder = SOFRCurveBuilder(curve_date=self.curve_date, sigma=sigma)
        daily_curve = builder.build(self.futures_filepath)
        
        # Create yield curve wrapper
        yield_curve = YieldCurve(daily_curve)
        
        # Price swaps and calculate errors
        errors = []
        for maturity, market_rate in self.market_swaps.items():
            swap = SOFRSwap(
                current_date=self.curve_date,
                maturity_str=maturity,
                fixed_rate=market_rate / 100,
                yield_curve=yield_curve,
                notional=100
            )
            
            fixed_pv = swap.price_fixed_leg()
            floating_pv = swap.price_floating_leg()
            error = floating_pv - fixed_pv
            errors.append(error)
        
        # Return sum of squared errors
        return sum(e**2 for e in errors)
    
    def calibrate(self, initial_sigma=0.007, bounds=(0.001, 0.008)):
        """
        Calibrate volatility to minimize swap pricing errors.
        
        Returns:
        --------
        dict : {'sigma': optimal_sigma, 'error': min_error}
        """
        result = minimize_scalar(
            self.calculate_swap_errors,
            bounds=bounds,
            method='bounded'
        )
        
        self.optimal_sigma = result.x
        
        # Rebuild curve with optimal sigma
        builder = SOFRCurveBuilder(curve_date=self.curve_date, sigma=self.optimal_sigma)
        self.optimal_curve = builder.build(self.futures_filepath)
        
        return {
            'sigma': self.optimal_sigma,
            'sigma_bps': self.optimal_sigma * 10000,
            'min_error': result.fun
        }

    def print_results(self):
        """Print swap pricing results using optimal sigma."""        
        yield_curve = YieldCurve(self.optimal_curve)
        
        print(f"\nOptimal Sigma: {self.optimal_sigma:.6f} ({self.optimal_sigma*10000:.2f} bps)")
        print(f"\n{'Maturity':<10} {'Market Rate':>12} {'Fixed PV':>12} {'Float PV':>12} {'Error':>12}")
        print("-" * 60)
        
        for maturity, market_rate in self.market_swaps.items():
            swap = SOFRSwap(
                current_date=self.curve_date,
                maturity_str=maturity,
                fixed_rate=market_rate / 100,
                yield_curve=yield_curve,
                notional=100
            )
            fixed_pv = swap.price_fixed_leg()
            floating_pv = swap.price_floating_leg()
            error = floating_pv - fixed_pv
            
            print(f"{maturity:<10} {market_rate:>12.5f} {fixed_pv:>12.6f} {floating_pv:>12.6f} {error:>12.6f}")