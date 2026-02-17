import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fixed_income.calibration.calibrate_nelson_siegel_svensson import CalibrateNelsonSiegelSvensson
from fixed_income.curves.nelson_siegel_svensson_spline import NelsonSiegelSvenssonSpline


BOND_INFO_PATH = r"C:\Users\anjal\Desktop\Fordham\Spring 2026\Interest Rate Derivatives\Homework\fixed_income\data\hw2_data\hw2_bond_info.csv"
PRICE_INFO_PATH = r"C:\Users\anjal\Desktop\Fordham\Spring 2026\Interest Rate Derivatives\Homework\fixed_income\data\hw2_data\hw2_bond_pricing.csv"

TAU1_FIXED = 4.0
TAU2_FIXED = 10.75


# -------------------------
# Load data
# -------------------------
cal = CalibrateNelsonSiegelSvensson(BOND_INFO_PATH, PRICE_INFO_PATH)

available_dates = sorted(cal.price_history["date"].unique()) # Get all available dates and sort them

# Choose a settlement date that is in the data 
SETTLEMENT_DATE = dt.date(2025, 11, 12)

# -------------------------
# Build bonds_today + clean_prices
# -------------------------
px_today = cal.price_history[cal.price_history["date"] == SETTLEMENT_DATE] # Keep the bond prices for the settlement date
price_map = dict(zip(px_today["cusip"], px_today["price"])) # Turn the table in to a dictionary "cusip": price for the settlement date

# bond_list is a list of FixedRateBond objects created using _load_data() method
# so every bond that was created in cal, we check if its cusip is in price_map 
# if so, we keep it for bonds_today
bonds_today = [b for b in cal.bond_list if b.ticker in price_map] # cusip is called ticker in FixedRateBond
clean_prices = [float(price_map[b.ticker]) for b in bonds_today] # get only the clean prices for the bonds on the settlement date

# -----------------------------
# Market PV (dirty) + DV01 list
# -----------------------------
pv_market = [] # list of dirty PVs
dv01_list = [] # list of DV01s calculated with yield from clean price

for b, clean in zip(bonds_today, clean_prices): # for each bond and its clean price 
    # market PV should match what pv_from_zero_curve returns - dirty 
    pv_mkt_dirty = float(clean) + float(b.accrued_interest(SETTLEMENT_DATE)) # calculate its dirty price
    pv_market.append(pv_mkt_dirty) # append dirty price to pv_market list

    y = b.price_to_yield_bisection(clean, SETTLEMENT_DATE) # get the yield from clean price using bisection
    dv01 = abs(float(b.DV01(y, SETTLEMENT_DATE))) # calculate DV01 using the yield
    dv01_list.append(max(dv01, 1e-10)) # append DV01 to dv01_list

# -------------------------
# Q3: NS Calibration (tau1 is fitted, tau2 irrelevant)
# -------------------------
ns_curve = NelsonSiegelSvenssonSpline(params=[0, 0, 0, 0, 4.0, 1.0])
ns_params, ns_res = ns_curve.fit(
    bond_list=bonds_today,
    pv_market_list=pv_market,
    dv01_list=dv01_list,
    settlement_date=SETTLEMENT_DATE,
    model="NS",
)

print("\nQ3 — Nelson–Siegel (NS) results")
print("Settlement date:", SETTLEMENT_DATE)
print("Parameters:", ns_params)
print("Objective value:", float(ns_res.fun))
print("Success:", bool(ns_res.success))


# -------------------------
# Q4: NSS Calibration (all params free)
# -------------------------
nss_curve = NelsonSiegelSvenssonSpline(params=[0, 0, 0, 0, TAU1_FIXED, TAU2_FIXED])
nss_params, nss_res = nss_curve.fit(
    bond_list=bonds_today,
    pv_market_list=pv_market,
    dv01_list=dv01_list,
    settlement_date=SETTLEMENT_DATE,
    model="NSS",
)

print("\nQ4 — Nelson–Siegel–Svensson (NSS, free taus) results")
print("Parameters:", nss_params)
print("Objective value:", float(nss_res.fun))
print("Success:", bool(nss_res.success))


# -------------------------
# Q5: NSS Calibration with fixed taus (tau1=4, tau2=10.75)
# -------------------------
nss_fix_curve = NelsonSiegelSvenssonSpline(params=[0, 0, 0, 0, TAU1_FIXED, TAU2_FIXED])
nss_fix_params, nss_fix_res = nss_fix_curve.fit(
    bond_list=bonds_today,
    pv_market_list=pv_market,
    dv01_list=dv01_list,
    settlement_date=SETTLEMENT_DATE,
    model="NSS",
    tau1=TAU1_FIXED,
    tau2=TAU2_FIXED,
)

print("\nQ5 — NSS (fixed taus) results")
print("tau1:", TAU1_FIXED, "tau2:", TAU2_FIXED)
print("Parameters:", nss_fix_params)
print("Objective value:", float(nss_fix_res.fun))
print("Success:", bool(nss_fix_res.success))


# -------------------------
# Curve tables (zero + DF)
# -------------------------
grid = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], float)

def curve_table(curve, grid):
    return pd.DataFrame({
        "t_years": grid,
        "zero_rate_cont": [curve.get_zero_rate(float(t)) for t in grid],
        "discount_factor": [curve.get_discount_factor(float(t)) for t in grid],
    })

print("\nCurve table (NS):\n", curve_table(ns_curve, grid))
print("\nCurve table (NSS free):\n", curve_table(nss_curve, grid))
print("\nCurve table (NSS fixed):\n", curve_table(nss_fix_curve, grid))


# -------------------------
# Pricing errors (dirty PV error)
# -------------------------
rows = []
for b, clean in zip(bonds_today, clean_prices):
    pv_mkt = float(clean) + float(b.accrued_interest(SETTLEMENT_DATE))
    pv_ns = float(b.pv_from_zero_curve(ns_curve, SETTLEMENT_DATE))
    pv_nss = float(b.pv_from_zero_curve(nss_curve, SETTLEMENT_DATE))
    pv_fix = float(b.pv_from_zero_curve(nss_fix_curve, SETTLEMENT_DATE))

    rows.append({
        "cusip": b.ticker,
        "pv_mkt_dirty": pv_mkt,
        "err_ns": pv_ns - pv_mkt,
        "err_nss_free": pv_nss - pv_mkt,
        "err_nss_fixed": pv_fix - pv_mkt,
    })

errs = pd.DataFrame(rows)

print("\nPricing error summary (mean abs error):")
print(errs[["err_ns", "err_nss_free", "err_nss_fixed"]].abs().mean())

print("\nPricing error summary (RMSE):")
print(np.sqrt((errs[["err_ns", "err_nss_free", "err_nss_fixed"]] ** 2).mean()))

# -------------------------
# Plot fitted zero curves
# -------------------------
maturity_grid = np.linspace(0.01, 30.0, 300)

plt.figure(figsize=(8,5))

plt.plot(
    maturity_grid,
    [ns_curve.get_zero_rate(t) for t in maturity_grid],
    label="NS"
)

plt.plot(
    maturity_grid,
    [nss_curve.get_zero_rate(t) for t in maturity_grid],
    label="NSS (free taus)"
)

plt.plot(
    maturity_grid,
    [nss_fix_curve.get_zero_rate(t) for t in maturity_grid],
    label="NSS (fixed taus)"
)

plt.xlabel("Maturity (years)")
plt.ylabel("Zero rate (continuous)")
plt.title(f"Fitted Zero Curves on {SETTLEMENT_DATE}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Q6: History + correlations + stability + objective comparison
# -------------------------

# Run history ONCE per method (Q3/Q4/Q5)
hist_ns = cal.run_history(model="NS")                            # Q3
hist_nss_free = cal.run_history(model="NSS", fix_lambdas=False)  # Q4
hist_nss_fix = cal.run_history(model="NSS", fix_lambdas=True)    # Q5 (fixed taus inside calibrator)

PARAM_NAMES = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]

def _params_df_from_history(hist: pd.DataFrame, param_names=PARAM_NAMES) -> pd.DataFrame:
    hist_ok = hist[hist["success"] == True].copy()
    hist_ok = hist_ok.sort_values("date")
    mat = np.vstack(hist_ok["params"].values)   # each params is length-6 array
    dfp = pd.DataFrame(mat, columns=param_names)
    dfp.insert(0, "date", hist_ok["date"].values)
    return dfp.set_index("date")

def _ok(hist: pd.DataFrame) -> pd.DataFrame:
    return hist[hist["success"] == True].sort_values("date")

# 1) Parameter time series
df_ns = _params_df_from_history(hist_ns)
df_free = _params_df_from_history(hist_nss_free)
df_fix = _params_df_from_history(hist_nss_fix)

# 2) Correlation matrices
print("\n=== Q6 Correlation matrix: NS ===\n", df_ns.corr())
print("\n=== Q6 Correlation matrix: NSS free taus ===\n", df_free.corr())
print("\n=== Q6 Correlation matrix: NSS fixed taus (4.0, 10.75) ===\n", df_fix.corr())

# 3) Standard deviation of parameters (stability)
std_table = pd.DataFrame({
    "NS": df_ns.std(),
    "NSS_free": df_free.std(),
    "NSS_fixed": df_fix.std(),
})
print("\n=== Q6 Parameter standard deviations (lower = more stable) ===\n", std_table)

# 4) Objective function history comparison
h_ns = _ok(hist_ns)
h_free = _ok(hist_nss_free)
h_fix = _ok(hist_nss_fix)

plt.figure()
plt.plot(h_ns["date"], h_ns["fun"], label="NS (Q3)")
plt.plot(h_free["date"], h_free["fun"], label="NSS free taus (Q4)")
plt.plot(h_fix["date"], h_fix["fun"], label="NSS fixed taus (Q5)")
plt.xticks(rotation=45)
plt.legend()
plt.title("Q6 Objective over time (all methods)")
plt.tight_layout()
plt.show()

# 5) Convergence summary 
def _summary(hist: pd.DataFrame) -> pd.Series:
    h = hist[hist["success"] == True]
    return pd.Series({
        "n_days_fit": len(h),
        "mean_fun": float(h["fun"].mean()),
        "median_fun": float(h["fun"].median()),
        "min_fun": float(h["fun"].min()),
        "max_fun": float(h["fun"].max()),
    })

summary_table = pd.DataFrame({
    "NS": _summary(hist_ns),
    "NSS_free": _summary(hist_nss_free),
    "NSS_fixed": _summary(hist_nss_fix),
})
print("\n=== Q6 Objective summary (lower is better fit) ===\n", summary_table)

# 6) show failure counts too
def _fail_count(hist: pd.DataFrame) -> int:
    return int((hist["success"] == False).sum())

print("\n=== Q6 Failure counts ===")
print("NS failures:", _fail_count(hist_ns))
print("NSS free failures:", _fail_count(hist_nss_free))
print("NSS fixed failures:", _fail_count(hist_nss_fix))

