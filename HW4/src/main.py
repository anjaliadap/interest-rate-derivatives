# main.py
from curve_builder import SOFRCurveBuilder
from yield_curve import YieldCurve
from calibrator import VolatilityCalibrator
import pandas as pd

CURVE_DATE = '2025-12-01'
FUTURES_FILE = r'C:\Users\anjal\Desktop\Fordham\Spring 2026\Interest Rate Derivatives\Homework\HW4\data\20251217_sofr_futures.csv'
SWAP_RATES_FILE = r'C:\Users\anjal\Desktop\Fordham\Spring 2026\Interest Rate Derivatives\Homework\HW4\data\20251217_sofr_swap_rates.csv'

# ============================================
# PART I: Build Curve (Q1-8)
# ============================================
print("="*60)
print("PART I: Building SOFR/OIS Curves")
print("="*60)

builder = SOFRCurveBuilder(curve_date=CURVE_DATE, sigma=0.007)
daily_curve = builder.build(FUTURES_FILE)
print(f"\nBuilt daily curve with {len(daily_curve)} days")

# ============================================
# PART II + III: Calibrate Volatility (Q9-11)
# ============================================
print("\n" + "="*60)
print("PART II: Calibrating Volatility & Pricing Swaps")
print("="*60)

# Load swap rates
swap_rates = pd.read_csv(SWAP_RATES_FILE)
swap_rates['maturity'] = swap_rates['ticker'].str[-1] + 'Y'
market_swaps = dict(zip(swap_rates['maturity'], swap_rates['rate']))

# Calibrate - this prices swaps internally
calibrator = VolatilityCalibrator(
    curve_date=CURVE_DATE,
    futures_filepath=FUTURES_FILE,
    market_swaps=market_swaps
)

result = calibrator.calibrate()

# Print results using calibrator's built-in reporting
calibrator.print_results()