import QuantLib as ql
US_CAL = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

# Internal mapping of day count conventions
_DAY_COUNT_CONVENTIONS = {
    "ACT/360": ql.Actual360,
    "ACT/365F": ql.Actual365Fixed,
    "ACT/ACT": lambda: ql.ActualActual(ql.ActualActual.ISDA),
    "30/360": lambda: ql.Thirty360(ql.Thirty360.USA),
}

# Change into QuantLib date format 
def get_ql_date(date: str) -> ql.Date:
    return ql.DateParser.parseFormatted(date, "%Y-%m-%d")

# Get maturity dates
def add_months(date: ql.Date, months: int = 6) -> ql.Date:
    unadjusted = date + ql.Period(months, ql.Months)
    return US_CAL.adjust(unadjusted, ql.Following)
    # ql.Period is used to add time periods to QuantLib dates

# Choose the day-count convention and calculate tau (year fraction)
def get_tau(start_date: ql.Date, end_date: ql.Date, dcc: str = "ACT/360") -> float:
    if dcc not in _DAY_COUNT_CONVENTIONS:
        raise ValueError(f"Day count convention '{dcc}' is not supported.")
    
    return _DAY_COUNT_CONVENTIONS[dcc]().yearFraction(start_date, end_date)

# Calculate the discount factors 
def get_discount_factor(rate:float, tau:float) -> float:
    return 1 / (1 + (rate*tau))

# Calculate the forward rate 
def get_forward_rate(rate_t:float, tau_t:float, rate_T:float, tau_T:float) -> float:
    if tau_T <= tau_t:
        raise ValueError("Maturity T must be greater than maturity t.")
    
    df_t = get_discount_factor(rate_t, tau_t)
    df_T = get_discount_factor(rate_T, tau_T)
    forward_rate = (df_t/df_T - 1) / (tau_T - tau_t)
    return forward_rate

# Enter date 
today_str = '2026-01-12'
today = get_ql_date(today_str)

rate_t = 5.125/100 # 6M spot rate
rate_T = 4.250/100 # 12M spot rate 

maturity_6M , maturity_12M = add_months(today, 6), add_months(today, 12)
tau_6M = get_tau(today, maturity_6M)
tau_12M = get_tau(today, maturity_12M)

forward_rate = get_forward_rate(
    rate_t = rate_t
    , tau_t = tau_6M
    , rate_T = rate_T
    , tau_T = tau_12M
)

print(f"The 6M-12M forward rate starting in 6M is: {forward_rate*100:.3f}%")