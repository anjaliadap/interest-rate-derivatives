# fixed_income

A Python library for fixed-income analytics, covering U.S. Treasury bond pricing, yield curve modeling, and Fed Funds futures analysis.

## Installation

```bash
pip install -e .
```

For development dependencies (pytest, openpyxl):

```bash
pip install -e ".[dev]"
```

## Library Structure

```
src/fixed_income/
├── dates/           Date utilities
│   ├── adjust.py        Business day adjustment, add_months, add_years, add_term
│   ├── calendars.py     FOMC meeting dates, Fed Funds settlement dates
│   ├── constants.py     CBOT month codes, IMM months
│   ├── convert.py       Date <-> epoch-days conversion
│   ├── daycount.py      Year fraction (ACT/360, ACT/365.25)
│   ├── imm.py           IMM date calculation, CBOT month code conversion
│   └── utils.py         Index search, term sorting, month splitting
│
├── instruments/     Financial instruments
│   ├── bond.py              FixedRateBond (pricing, YTM, DV01, convexity, CPN01)
│   └── fed_funds_futures.py FedFundsFutures (FOMC rate extraction via bootstrap/lstsq)
│
├── curves/          Yield curve models
│   ├── zero_curve.py                    ZeroCurve abstract base class
│   ├── dummy_zero_curve.py              Flat-rate curve for testing
│   ├── nelson_siegel_svensson_spline.py Nelson-Siegel / Nelson-Siegel-Svensson model
│   └── ffer_curve.py                    Fed Funds Effective Rate front-end curve
│
├── calibration/     Curve fitting
│   └── calibrate_nelson_siegel_svensson.py  Fit NS/NSS curves to bond price data
│
└── market_data/     Data loaders
    └── bloomberg_csv.py  Bloomberg Excel file reader for Fed Funds futures
```

## Usage

### Bond Pricing

```python
import datetime
from fixed_income.instruments import FixedRateBond

bond = FixedRateBond(
    ticker="912810RP",
    coupon_rate=0.03,
    maturity_date=datetime.date(2045, 11, 15),
    issue_date=datetime.date(2025, 12, 1),
    dated_date=datetime.date(2025, 11, 15),
)

settlement = datetime.date(2025, 12, 1)

# Quoted (clean) price from yield
price = bond.yield_to_quoted_price(0.045, settlement)

# Yield from market price
ytm = bond.price_to_yield_bisection(77.46, settlement)

# Risk measures
dv01 = bond.DV01(ytm, settlement)
conv = bond.convexity(ytm, settlement)
ai = bond.accrued_interest(settlement)
```

### Yield Curves

```python
from fixed_income.curves import DummyZeroCurve, NelsonSiegelSvenssonSpline

# Flat curve for testing
flat = DummyZeroCurve(0.04)
df = flat.get_discount_factor(5.0)

# Nelson-Siegel-Svensson curve
params = [0.04, -0.02, 0.02, 0.01, 4.0, 10.0]  # [b0, b1, b2, b3, tau1, tau2]
nss = NelsonSiegelSvenssonSpline(params)
zero_rate = nss.get_zero_rate(10.0)
fwd_rate = nss.get_forward_rate(5.0, 10.0)

# Price a bond off a zero curve
pv = bond.pv_from_zero_curve(nss, settlement)
```

### Date Utilities

```python
import datetime
from fixed_income.dates import (
    add_months, add_term, next_business_day,
    get_next_IMM_Date, year_fraction, get_fomc_meeting_dates,
)

# Business day arithmetic
nbd = next_business_day(datetime.date(2025, 12, 25))

# Term arithmetic
maturity = add_term(datetime.date(2025, 1, 1), "5Y")

# IMM dates
imm = get_next_IMM_Date(datetime.date(2025, 10, 1))

# FOMC calendar
fomc_dates = get_fomc_meeting_dates(2025)
```

### Curve Calibration

```python
from fixed_income.calibration import CalibrateNelsonSiegelSvensson

cal = CalibrateNelsonSiegelSvensson(
    bond_info_path="data/hw2_data/hw2_bond_info.csv",
    price_info_path="data/hw2_data/hw2_bond_pricing.csv",
)

# Fit a Nelson-Siegel curve for a single date
params, result = cal.fit_spline(datetime.date(2025, 11, 12), model="NS")

# Run across all dates in price history
history = cal.run_history(model="NSS", fix_lambdas=True)
```

## Dependencies

- numpy
- pandas
- scipy
- holidays
