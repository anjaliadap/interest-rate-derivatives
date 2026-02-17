
# USTreasuryBond.py
# A class to model U.S. Treasury Bonds with methods for pricing, yield calculation, accrued interest, DV01, and convexity.
# DISCLAIMER: THERE WILL BE BUGS IN THIS CODE.
# BUGS OF ALL KINDS ARE LIKELY PRESENT, INCLUDING LOGICAL, MATHEMATICAL, PROGRAMMING ERRORS AND STYLE ERRORS.
# BUGS MAY CAUSE INCORRECT RESULTS.
# THERE WILL BE NO SUPPORT FOR THIS CODE. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
# USE AT YOUR OWN RISK.
# FEEL FREE TO MODIFY AND IMPROVE THE CODE.
# BY USING THIS CODE, YOU AGREE TO THE TERMS OF THIS DISCLAIMER.
# THIS CODE IS PROVIDED FOR EDUCATIONAL PURPOSES ONLY.
# DO NOT USE THIS CODE FOR REAL TRADING OR INVESTMENT DECISIONS.
# THE AUTHOR IS NOT RESPONSIBLE FOR ANY LOSSES OR DAMAGES ARISING FROM THE USE OF THIS CODE.
# DO NOT DISTRIBUTE THIS CODE WITHOUT PERMISSION FROM THE AUTHOR.
# FOR PERSONAL USE ONLY


import datetime
from DateUtilities import add_months
from pandas.tseries.holiday import USFederalHolidayCalendar
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# START CLASS ---------------------------------------------------------------------------------------------------------------------------------------
class USTreasuryBond:

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self
                 , ticker
                 , issue_date
                 , dated_date
                 , maturity_date
                 , coupon_rate
                 , face_value=100):
        
        self.ticker = ticker
        self.issue_date = issue_date
        self.dated_date = dated_date
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate # Annual coupon rate as a decimal
        self.face_value = face_value

        # List of tuples (start_date, end_date) for each coupon period
        self.coupon_periods = self._generate_coupon_period_dates(dated_date, maturity_date) 

        # List of coupon period dates from dated date to maturity date
        self.coupon_period_dates = [dated_date] + [end_date for (start_date, end_date) in self.coupon_periods] 

    #---------------------------------------------------------------------------------------------------------------------------------------
    def _generate_coupon_period_dates(self, start_date, end_date):
        """
        Generate coupon payment dates between start_date and end_date, which are both coupon period dates

        Parameters:
        start_date (datetime.date): The date from which to start generating coupon dates. 
                                : This is also called the dated date, which is the date from which interest starts accruing around the issue date. 
        end_date (datetime.date): The maturity date of the bond.

        Returns:
        list: A list of coupon period date tuples (start_date, end_date).
        """

        coupon_periods = []  # Stores each coupon period as (start, end)
        current_period_start_date = start_date  # First period starts at dated date (date fro which interest accrues)

        # Dummy calendar with no holidays (add_months expects .holidays)
        class no_calendar:
            def __init__(self):
                self.holidays = []

        cal_no = no_calendar()

        # First coupon period end date (6 months after start)
        current_period_end_date = add_months(
            current_period_start_date,
            num_months=6,
            cal=cal_no,
            day_convention="None",
            eom=True
        )

        # Add the first coupon period
        current_period = (current_period_start_date, current_period_end_date)
        coupon_periods.append(current_period)

        # Generate remaining coupon periods until maturity
        while current_period_end_date < end_date:
            current_period_start_date = current_period_end_date # Next period starts where the last one ended
            # Next coupon period end date (6 months after start)
            current_period_end_date = add_months(
                current_period_start_date,
                num_months=6,
                cal=cal_no, # the class defined above with no holidays was created because add_months expects a calendar with a .holidays attribute
                day_convention="None",
                eom=True
            )
            current_period = (current_period_start_date, current_period_end_date)
            coupon_periods.append(current_period)

        return coupon_periods

    #---------------------------------------------------------------------------------------------------------------------------------------
    def yield_to_pv(self, yield_rate, settlement_date): # payments scale by face value, yields and coupon rates are decimal.
        """
        Calculate the dirty price (present value) of the USTreasuryBond given a yield rate and settlement date.

        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.

        Returns:
        float: The price of the bond.
        """
        pv = 0.0
        period_number = 0
        year_fraction = None

        for period_start, period_end in self.coupon_periods:
            if period_end > settlement_date: # Check if the settlement date is within this coupon period
                if year_fraction == None: # If yes, calculate the year fraction for this period only
                    year_fraction = (period_end - settlement_date) / (period_end - period_start) # Ideally it will be x / 180 days for 6 month periods
                
                coupon_payment = (self.coupon_rate / 2) * self.face_value # Semi-annual coupon payment
                pv += coupon_payment / ((1 + yield_rate / 2) ** (period_number + year_fraction)) 
                # Period number refers to how much time has already passed. If 1 year has passed, period number is 2 because there are 2 coupon payments per year.
                # The current payment is then discounted by period_number + year_fraction, where year_fraction is the fraction of the current coupon period remaining.
                period_number += 1
        
        pv += self.face_value / ((1 + yield_rate / 2) ** (period_number-1 + year_fraction)) # Discount the face value payment at maturity
        
        return pv
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    def accrued_interest(self, settlement_date):
        """
        Calculate the accrued interest of the USTreasuryBond as of the settlement date.

        Parameters:
        settlement_date (datetime.date): The date on which the bond is being priced.

        Returns:
        float: The accrued interest.
        """
        last_coupon_date = None
        next_coupon_date = None

        for period_start, period_end in self.coupon_periods:
            if period_end > settlement_date: # Check if the settlement date is within this coupon period
                next_coupon_date = period_end 
                last_coupon_date = period_start
                break

        if last_coupon_date is None or next_coupon_date is None:
            return 0.0 # No accrued interest if settlement date is outside coupon periods
        
        accrued_days = (settlement_date - last_coupon_date).days # Get the number of days between the last coupon date and settlement date
        total_days = (next_coupon_date - last_coupon_date).days # Get the total number of days in the coupon period
        accrued_interest = (self.coupon_rate / 2) * self.face_value * (accrued_days / total_days) # Accrued interest = semi-annual coupon * (accrued days / total days in period)
        
        return accrued_interest

    #---------------------------------------------------------------------------------------------------------------------------------------
    def yield_to_price(self, yield_rate, settlement_date):
        """
        Calculate the quoted price of the USTreasuryBond, which excludes accrued interest. 

        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.

        Returns:
        float: The quoted price of the bond, which excludes accrued interest.
        """
        pv = self.yield_to_pv(yield_rate, settlement_date) # Get the dirty price (present value)
        accrued_interest = self.accrued_interest(settlement_date) # Get the accrued interest
        clean_price = pv - accrued_interest # Get the quoted price (clean price) by subtracting accrued interest from dirty price
        
        return clean_price
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    def price_to_yield(self, market_price, settlement_date, tol=1e-6, max_iter=100):
        """
        Calculate the yield to maturity of the USTreasuryBond given a market price and settlement date.

        Parameters:
        market_price (float): The market price of the bond.
        settlement_date (datetime.date): The date on which the bond is being priced.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.

        Returns:
        float: The yield to maturity (as a decimal).
        """

        # The formula for pv is a summation that cannot be inverted analytically to get yield.
        # Therefore, we use a numerical method (bisection method here) to find the yield that gives the desired market price.

        lower_bound = 0.0
        upper_bound = 1.0
        
        for _ in range(max_iter): # bisection method. Can be improved with newton-raphson or secant method or solvers from scipy
            
            mid_yield = (lower_bound + upper_bound) / 2
            price = self.yield_to_price(mid_yield, settlement_date)
            
            if abs(price - market_price) < tol:
                return mid_yield
            if price < market_price:
                upper_bound = mid_yield
            else:
                lower_bound = mid_yield
        
        raise ValueError("Yield to maturity calculation did not converge")

    #---------------------------------------------------------------------------------------------------------------------------------------
    def price_to_coupon(self, price, base_yield, settlement_date, tol=1e-6, max_iter=100): 
        """
        Calculate the coupon rate of the USTreasuryBond given a market price and settlement date.

        Parameters:
        market_price (float): The market price of the bond.
        settlement_date (datetime.date): The date on which the bond is being priced.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.

        Returns:
        float: The coupon rate (as a decimal).
        """
        test_bond = USTreasuryBond(self.ticker, self.issue_date, self.dated_date, self.maturity_date, self.coupon_rate, self.face_value)

        lower_bound = 0.0 # Coupon rate cannot be negative
        upper_bound = 0.5 # Arbitrary upper bound for coupon rate (50%)
        
        for _ in range(max_iter):
            mid_coupon = (lower_bound + upper_bound) / 2 # Test if midpoint coupon rate gives the desired price
            test_bond.coupon_rate = mid_coupon
            mid_price = test_bond.yield_to_price(base_yield, settlement_date) # Price of bond with mid_coupon at base_yield
            
            if abs(price - mid_price) < tol: # If price is close enough to mid_price, return mid_coupon as the coupon rate
                return mid_coupon
            if price < mid_price: # if price is less than mid_price, need to lower coupon
                upper_bound = mid_coupon
            else: # if price is greater than mid_price, need to increase coupon
                lower_bound = mid_coupon
        
        raise ValueError("Coupon rate calculation did not converge")
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    #----- QUESTION 3 -----
    def pv_from_zcb_yield_curve(self, zcb_yield_curve, settlement_date): # Don't pass the argument for zcb_yield_curve in the function definiton. 
                                                                         # Pass it inside the definition of pv_from_zcb_yield_curve when calling the function.
                                                                         # Inside the function definiton of pv_from_zcb_yield_curve, zcb_yield_curve becomes a variable whose value is a function.
                                                                         # Passing a function != Calling a function.
        """
        Calculate the price of the USTreasuryBond using a zero-coupon bond yield curve.

        Parameters:
        zcb_yield_curve (function): A function that takes time to maturity (in years) and returns the zero-coupon yield.
        settlement_date (datetime.date): The date on which the bond is being priced.

        Returns:
        float: The pv of the bond. 
        IMPORTANT: pv is defined as the sum of the present values of all future cash flows (coupons that are paid do not contribute), discounted using the zero-coupon yield curve.
        """
        pv = 0.0 # Initialize present value to be added to this variable 

        for period_start, period_end in self.coupon_periods:
            if period_end > settlement_date: # Check if the settlement date is within this coupon period
                coupon_payment = (self.coupon_rate / 2) * self.face_value # Semi-annual coupon payment
                
                # Time to maturity in years for this coupon payment
                time_to_maturity = (period_end - settlement_date).days / 365.0 
                
                # Get the zero-coupon yield for this time to maturity
                zcb_yield = zcb_yield_curve(time_to_maturity)
                
                # Discount the coupon payment using the zero-coupon yield
                pv += coupon_payment / ((1 + zcb_yield / 2) ** (time_to_maturity * 2)) # Semi-annual compounding

        # Discount the face value payment at maturity
        time_to_maturity = (self.maturity_date - settlement_date).days / 365.0
        zcb_yield = zcb_yield_curve(time_to_maturity)
        pv += self.face_value / ((1 + zcb_yield / 2) ** (time_to_maturity * 2)) # Semi-annual compounding
        
        return pv
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    def CPN01(self, yield_rate, settlement_date, delta_yield=0.0001): #DV01 is positive, negative of the derivative of price wrt yield
        """
        Calculate the DV01 (Dollar Value of 01) of the USTreasuryBond.
        CPN01 measures how sensitive the bond price is to a 1bp change in the coupon rate.

        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        delta_yield (float): The change in yield to calculate DV01.

        Returns:
        float: The DV01 of the bond.
        """
        bond_cpn_up = USTreasuryBond("", self.issue_date, self.dated_date, self.maturity_date, self.coupon_rate + 0.0001, self.face_value)
        price_up = bond_cpn_up.price_quoted(yield_rate, settlement_date)
        
        bond_cpn_down = USTreasuryBond("", self.issue_date, self.dated_date, self.maturity_date, self.coupon_rate - 0.0001, self.face_value)
        price_down = bond_cpn_down.price_quoted(yield_rate, settlement_date)
        
        cpn01 = (price_up - price_down) / 2 # Central price difference formula
        
        return cpn01

    #---------------------------------------------------------------------------------------------------------------------------------------
    def DV01(self, yield_rate, settlement_date, delta_yield=0.0001): #DV01 is positive, negative of the derivative of price wrt yield
        """
        Calculate the DV01 (Dollar Value of 01) of the USTreasuryBond.
        DV01 is the change in a bond's price for a 1 bp increase in yield. 

        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        delta_yield (float): The change in yield to calculate DV01.

        Returns:
        float: The DV01 of the bond.
        """
        price_up = self.yield_to_price(yield_rate + delta_yield, settlement_date)
        price_down = self.yield_to_price(yield_rate - delta_yield, settlement_date)
        
        dv01 = (price_down - price_up) / 2 # Central price difference formula 
        
        return dv01
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    def convexity(self, yield_rate, settlement_date, delta_yield=0.0001):
        """
        Calculate the convexity of the USTreasuryBond.
        Convexity measures how the DV01 of a bond changes as yield changes. 

        Parameters:
        yield_rate (float): The annual yield rate (as a decimal).
        settlement_date (datetime.date): The date on which the bond is being priced.
        delta_yield (float): The change in yield to calculate convexity.

        Returns:
        float: The convexity of the bond.
        """
        price_up = self.yield_to_price(yield_rate + delta_yield, settlement_date)
        price_down = self.yield_to_price(yield_rate - delta_yield, settlement_date)
        price_current = self.yield_to_price(yield_rate, settlement_date)
        
        convexity = (price_up + price_down - 2 * price_current)   
        
        return convexity

# END CLASS ---------------------------------------------------------------------------------------------------------------------------------------

#----- Part of QUESTION 3-----
def zcb_yield_curve(time_to_maturity):
    """
    Example zero-coupon bond yield curve function.
    At present this function returns dummy values of yields based on time to maturity.

    Parameters:
    time_to_maturity (float): Time to maturity in years.

    Returns: 
    float: The zero-coupon yield (as a decimal).
    """
    if time_to_maturity <= 1:
        return 0.03 # 3% for <= 1 year
    elif time_to_maturity <= 5:
        return 0.04 # 4% for > 1 year and <= 5 years
    else:
        return 0.05 # 5% for > 5 years
    
#---------------------------------------------------------------------------------------------------------------------------------------
#---- Part of QUESTION 4 -----
def bond_plotter(bond, settlement_date, bond_price=100.00, name=None):
    """
    This function plots bond price, bond DV01 and bond convexity as a function of yield.
    
    Parameters:
    bond: An instance of the USTreasuryBond class.
    settlement_date (datetime.date): The date on which the bond is being priced.
    bond_price (float): The market price of the bond.
    name (str): A label for the bond in the plots.
    """

    # calculate the yield corresponding to the given bond price
    base_yield = bond.price_to_yield(bond_price, settlement_date) # For bond price of 100.00, this is the par yield
    base_dv01 = bond.DV01(base_yield, settlement_date)
    base_convexity = bond.convexity(base_yield, settlement_date)

    # Generate yield values around the base yield in the range of base_yield +/- 300 bps = 3%
    yield_range = np.linspace(base_yield - 0.03, base_yield + 0.03, 100)
    delta_y_bp = (yield_range - base_yield) / 0.0001 # Yield changes in basis points from base yield

    # Calculate all the bond prices, DV01s and convexities for these yield values
    range_prices = [bond.yield_to_price(y, settlement_date) for y in yield_range]
    range_dv01s = [bond.DV01(y, settlement_date) for y in yield_range]
    range_convexities = [bond.convexity(y, settlement_date) for y in yield_range]

    # Calculate linear prices using DV01 approximation
    linear_prices = bond_price - base_dv01 * delta_y_bp
    # Calculate linear prices using DV01 and Convexity approximation 
    quadratic_prices = bond_price - base_dv01 * delta_y_bp + 0.5 * base_convexity * (delta_y_bp **2)

    # Error with DV01 linear approximation
    error_dv01 = np.array(range_prices) - np.array(linear_prices)
    error_dv01_convexity = np.array(range_prices) - np.array(quadratic_prices)

    # Plot a grid of 4 plots: Price vs Yield, DV01 vs Yield, Convexity vs Yield, Error in Price using DV01 only vs Yield
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} (+/- 300 bp around par)", fontsize=14)

    axs[0, 0].plot(yield_range * 100, range_prices, label="True Price")
    axs[0, 0].plot(yield_range * 100, linear_prices, linestyle="--", label="Linear (DV01 at y0)")
    axs[0, 0].plot(yield_range * 100, quadratic_prices, linestyle=":", label="Quadratic (DV01 + Convexity at y0)")
    axs[0, 0].axvline(base_yield * 100, color="r", linestyle="--", label="Base Yield")
    axs[0, 0].set_title("Price vs Yield")
    axs[0, 0].set_xlabel("Yield (%)")
    axs[0, 0].set_ylabel("Price (per $100)")  
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(yield_range * 100, range_dv01s, label="DV01")
    axs[0, 1].axvline(base_yield * 100, color="r", linestyle="--", label="Base Yield")
    axs[0, 1].set_title("DV01 vs Yield")
    axs[0, 1].set_xlabel("Yield (%)")
    axs[0, 1].set_ylabel("DV01 (price per 1bp)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(yield_range * 100, range_convexities, label="Convexity")
    axs[1, 0].axvline(base_yield * 100, color="r", linestyle="--", label="Base Yield")
    axs[1, 0].set_title("Convexity vs Yield")
    axs[1, 0].set_xlabel("Yield (%)")
    axs[1, 0].set_ylabel("Convexity (per (1 bp)^2)")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(yield_range * 100, error_dv01, label="Error = True - Linear")
    axs[1, 1].plot(yield_range * 100, error_dv01_convexity, label="Error = True - Quadratic")
    axs[1, 1].axhline(0, color="k", linewidth=1)
    axs[1, 1].axvline(base_yield * 100, color="r", linestyle="--", label="Base Yield")
    axs[1, 1].set_title("Price Error")
    axs[1, 1].set_xlabel("Yield (%)")
    axs[1, 1].set_ylabel("Error (price)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------
#----- Part of QUESTION 5 -----
def simulate_yield_diffusion(y0, sigma=0.01, T_years=0.5, steps_per_year=252, n_paths=1000, seed=42):
    """
    Simulate yield paths under: dy = sigma * dW

    Parameters:
    y0: initial yield
    sigma: volatility of yield
    T_years: time horizon in years
    steps_per_year: number of steps per year for business days 
    n_paths: number of paths to simulate
    seed: random seed for reproducibility

    Result:
    y_paths: yield paths for different 
    """
    rnge = np.random.default_rng(seed)

    n_steps = int(T_years * steps_per_year)
    dt = T_years / n_steps

    # Brownian increments 
    Z = rnge.standard_normal(size=(n_paths, n_steps))
    dy = sigma * np.sqrt(dt) * Z # dy = sigma * dW, dW = sqrt(dt) * Z

    # Yield Paths
    y_paths = y0 + np.cumsum(dy, axis=1)

    # Add t=0 column so shape becomes (n_paths, n_steps+1)
    y0_col = np.full((n_paths, 1), y0)
    y_paths = np.hstack([y0_col, y_paths])
    y_terminal = y_paths[:, -1]

    # Theoretical disctribution
    mu_T = y0
    std_T = sigma * np.sqrt(T_years)

    x = np.linspace(mu_T - 4*std_T, mu_T + 4*std_T, 100)
    pdf_T = (1 / (std_T * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_T) / std_T) ** 2)

    # Plotting
    plt.figure(figsize=(12, 6))
    for i in range(min(n_paths, 50)): # Plot only first 50 paths for clarity
        plt.plot(np.linspace(0, T_years, n_steps+1), y_paths[i], color='lightgray', alpha=0.5)
    plt.plot(np.linspace(0, T_years, n_steps+1), np.mean(y_paths, axis=0), color='blue', label='Mean Yield Path', linewidth=2)
    plt.title('Simulated Yield Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Yield')
    plt.grid()
    plt.legend()
    plt.show()

    # Terminal Yield Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(y_terminal, bins=30, density=True, alpha=0.6, label='Simulated Terminal Yields')
    plt.plot(x, pdf_T, 'r-', lw=2, label='Theoretical Distribution')
    plt.title('Terminal Yield Distribution')
    plt.xlabel('Yield at T')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()

    return y_paths

#---------------------------------------------------------------------------------------------------------------------------------------
#----- Part of QUESTION 6 -----
def pnl_error_from_simulated_yields(bond, yield_paths, start_date):
    """
    For each day in each simulation path:
      - Convert simulated yield -> simulated clean price
      - Compute 1-day PnL = P_t - P_{t-1}
      - Explain PnL using DV01 only (Taylor 1st order)
      - Explain PnL using DV01 + convexity (Taylor 2nd order)
      - Return error distributions

    Parameters:
    bond : USTreasuryBond object 
    yield_paths : np.ndarray
        Simulated yields in DECIMAL, shape (n_paths, n_steps+1)
        Example: yield_paths[i, t] is yield on day t for path i
    start_date : datetime.date
        The settlement date for t=0 

    Returns
    results : dict of np.ndarray
        Flattened arrays across all paths and days:
          - pnl_actual
          - pnl_dv01
          - pnl_dv01_conv
          - err_dv01
          - err_dv01_conv
          - dy_bp
    """

    yield_paths = np.asarray(yield_paths)
    if yield_paths.ndim != 2:
        raise ValueError("yield_paths must be 2D: shape (n_paths, n_steps+1)")

    n_paths, n_cols = yield_paths.shape
    n_steps = n_cols - 1
    if n_steps < 1:
        raise ValueError("yield_paths must have at least 2 columns (n_steps+1 >= 2)")

    # Build the date grid 
    date_index = pd.bdate_range(start=start_date, periods=n_cols).date

    bp = 1e-4  # 1bp in decimal yield terms

    pnl_actual_list = []
    pnl_dv01_list = []
    pnl_dv01_conv_list = []
    err_dv01_list = []
    err_dv01_conv_list = []
    dy_bp_list = []

    # Loop through paths
    for i in range(n_paths):

        # Price all dates in this path (pricing uses the date as settlement_date)
        prices = np.zeros(n_cols)
        for t in range(n_cols):
            y_t = float(yield_paths[i, t])
            prices[t] = bond.yield_to_price(y_t, date_index[t])

        # compute daily PnL and explained PnL
        for t in range(1, n_cols):
            y_prev = float(yield_paths[i, t-1])
            y_curr = float(yield_paths[i, t])
            dy = y_curr - y_prev
            dy_in_bp = dy / bp  # convert yield change to bps units

            P_prev = prices[t-1]
            P_curr = prices[t]
            pnl_actual = P_curr - P_prev

            # Risk measures at previous day 
            dv01_prev = bond.DV01(y_prev, date_index[t-1]) # $ per 1bp
            cx_prev   = bond.convexity(y_prev, date_index[t-1]) # $ per (1bp)^2 

            # DV01-only explained PnL:
            # P(y+dy) ≈ P(y) - DV01 * dy_bp  => pnl ≈ - DV01 * dy_bp
            pnl_dv01 = -dv01_prev * dy_in_bp

            # DV01 + convexity explained PnL (2nd order Taylor):
            # P(y+dy) ≈ P(y) - DV01*k + 0.5*Conv*k^2, with k=dy_bp
            pnl_dv01_conv = -dv01_prev * dy_in_bp + 0.5 * cx_prev * (dy_in_bp ** 2)

            err_dv01 = pnl_actual - pnl_dv01
            err_dv01_conv = pnl_actual - pnl_dv01_conv

            pnl_actual_list.append(pnl_actual)
            pnl_dv01_list.append(pnl_dv01)
            pnl_dv01_conv_list.append(pnl_dv01_conv)
            err_dv01_list.append(err_dv01)
            err_dv01_conv_list.append(err_dv01_conv)
            dy_bp_list.append(dy_in_bp)

    results = {
        "pnl_actual": np.array(pnl_actual_list),
        "pnl_dv01": np.array(pnl_dv01_list),
        "pnl_dv01_conv": np.array(pnl_dv01_conv_list),
        "err_dv01": np.array(err_dv01_list),
        "err_dv01_conv": np.array(err_dv01_conv_list),
        "dy_bp": np.array(dy_bp_list),
    }

    return results

def plot_error_histograms(results, title_prefix="CT10"):
    """
    Simple helper for the plots:
    """
    err1 = results["err_dv01"]
    err2 = results["err_dv01_conv"]

    plt.figure(figsize=(10, 5))
    plt.hist(err1, bins=60, alpha=0.6, density=True, label="Error: Actual - DV01")
    plt.hist(err2, bins=60, alpha=0.6, density=True, label="Error: Actual - (DV01+Conv)")
    plt.title(f"{title_prefix}: Error Distributions")
    plt.xlabel("PnL Error (price units per $100)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------
#----- Part of QUESTION 7 -----
def portfolio_dv01(bonds, notionals_mm, settlement_date):
    """
    Returns total portfolio DV01 in DOLLARS per 1bp.
    notionals_mm are in $MM (e.g. 1000 = $1000MM).
    """
    total = 0.0
    for bond, notional_mm in zip(bonds, notionals_mm):
        ytm = bond.price_to_yield(100.0, settlement_date)
        dv01_per_100 = bond.DV01(ytm, settlement_date)  # price change per 1bp per $100 face
        total += dv01_per_100 * (notional_mm * 1_000_000) / 100.0
    return total

def hedge_notional(portfolio_dv01_dollars, bond10, settlement_date):
    """
    Returns hedge notional in $MM of the 10Y needed to neutralize portfolio DV01.
    portfolio_dv01_dollars must be in DOLLARS per 1bp.
    """
    y10 = bond10.price_to_yield(100.0, settlement_date)
    dv01_10_per_100 = bond10.DV01(y10, settlement_date)

    # DV01 of 10Y per $1MM notional (dollars per 1bp)
    dv01_10_per_1mm = dv01_10_per_100 * (1_000_000 / 100.0)

    # Hedge notional in $MM
    return -portfolio_dv01_dollars / dv01_10_per_1mm

def scenario_pnl(bonds, notionals_mm, shocks_bp, settlement_date):
    """
    Returns instantaneous PnL in DOLLARS using DV01 approximation:
      dP ≈ -DV01 * shock_bp
    """
    pnl = 0.0
    for bond, notional_mm, shock_bp in zip(bonds, notionals_mm, shocks_bp):
        y0 = bond.price_to_yield(100.0, settlement_date)
        dv01_per_100 = bond.DV01(y0, settlement_date)
        pnl += -dv01_per_100 * shock_bp * (notional_mm * 1_000_000) / 100.0
    return pnl
#---------------------------------------------------------------------------------------------------------------------------------------

def main():
    #----- QUESTION 2 -----
    print("---------------- QUESTION 2 ----------------\n")
    price = 100.00
    issue_date = datetime.date(2025, 11, 17)
    dated_date = datetime.date(2025, 11, 15)
    maturity_date = datetime.date(2035, 11, 15)
    coupon_rate = 0.04 # 4%
    bond = USTreasuryBond('91282CPJ4', issue_date, dated_date, maturity_date, coupon_rate)

    settlement_date = datetime.date(2026, 1, 13)
       
    accrued_interest = bond.accrued_interest(settlement_date)
    ytm = bond.price_to_yield(price, settlement_date)
    price_from_yield = bond.yield_to_price(ytm, settlement_date)
    dv01 = bond.DV01(ytm, settlement_date)
    convexity = bond.convexity(ytm, settlement_date)

    print("Bond Results (per $100 notional):")
    print(f"Quoted Price: {price:.6f}")
    print(f"Accrued Interest: {accrued_interest:.6f}")
    print(f"Yield to Maturity: {ytm*100:.6f}%")

    print(f"DV01: {dv01:.6f} per 1 bp change in yield")
    dv01_1mm = dv01 / 100 * 1_000_000
    print(f"DV01 (per $1mm): {dv01_1mm:.2f}")

    print(f"Convexity: {convexity:.6f} per (1 bp)^2 change in yield")
    convexity_bbg = (convexity / price) / (0.0001**2) * 0.01
    print(f"Convexity (Following Bloomberg): {convexity_bbg:.3f}")
    print()

    #----- QUESTION 4 -----
    print("---------------- QUESTION 4 ----------------\n")

    # --- CT2: 3 3/8% 12/31/2027 ---
    bond2 = USTreasuryBond(
        ticker="CT2",
        issue_date=datetime.date(2025, 12, 31),
        dated_date=datetime.date(2025, 12, 31),
        maturity_date=datetime.date(2027, 12, 31),
        coupon_rate=0.03375
    )

    # --- CT5: 3 5/8% 12/31/2030 ---
    bond5 = USTreasuryBond(
        ticker="CT5",
        issue_date=datetime.date(2025, 12, 31),
        dated_date=datetime.date(2025, 12, 31),
        maturity_date=datetime.date(2030, 12, 31),
        coupon_rate=0.03625
    )

    # --- CT10: 4% 11/15/2035 ---
    bond10 = USTreasuryBond(
        ticker="CT10",
        issue_date=datetime.date(2025, 11, 17),
        dated_date=datetime.date(2025, 11, 15),
        maturity_date=datetime.date(2035, 11, 15),
        coupon_rate=0.04
    )

    # --- CT30: 4 5/8% 11/15/2055 ---
    bond30 = USTreasuryBond(
        ticker="CT30",
        issue_date=datetime.date(2025, 11, 17),
        dated_date=datetime.date(2025, 11, 15),
        maturity_date=datetime.date(2055, 11, 15),
        coupon_rate=0.04625
    )

    # Plot bonds
    bond_plotter(bond2, settlement_date, name="CT2")
    bond_plotter(bond5, settlement_date, name="CT5")
    bond_plotter(bond10, settlement_date, name="CT10")
    bond_plotter(bond30, settlement_date, name="CT30")

    #----- QUESTION 5 -----
    print("---------------- QUESTION 5 ----------------\n")
    y0 = 0.03999461  # CT10 coupon = par yield

    yield_paths = simulate_yield_diffusion(y0=y0
                             , sigma=0.01 # 100 bps
                             , T_years=0.5 # six months 
                             , steps_per_year=252
                             , n_paths=10000
                             , seed=42)
    
    #----- Question 6 -----
    print("---------------- QUESTION 6 ----------------\n")
    results = pnl_error_from_simulated_yields(bond10, yield_paths, settlement_date)
    plot_error_histograms(results, title_prefix="CT10")

    print("The DV01-only PnL attribution results in a wide and skewed error distribution due to nonlinear price–yield effects.\nIncluding convexity significantly reduces the magnitude and dispersion of the error, producing a tighter and more symmetric distribution centered near zero.\n")

    #----- Question 7 -----
    print("---------------- QUESTION 7 ----------------\n")

    # --- CT3: 3 1/2% 01/15/2029 ---
    bond3 = USTreasuryBond(
        ticker="CT3",
        issue_date=datetime.date(2026, 1, 15),
        dated_date=datetime.date(2026, 1, 15),
        maturity_date=datetime.date(2029, 1, 15),
        coupon_rate=0.035
    )

    # --- CT7: 3 7/8% 12/31/2032 ---
    bond7 = USTreasuryBond(
        ticker="CT7",
        issue_date=datetime.date(2025, 12, 31),
        dated_date=datetime.date(2025, 12, 31),
        maturity_date=datetime.date(2032, 12, 31),
        coupon_rate=0.03875
    )

    bonds = [
        bond2,   # 2Y
        bond3,   # 3Y
        bond5,   # 5Y
        bond7,   # 7Y
        bond10,  # 10Y
        bond30   # 30Y
    ]
    notionals = [
        1000,  # 2Y ($MM)
        500,   # 3Y
        -100,  # 5Y
        600,   # 7Y
        -300,  # 10Y
        200    # 30Y
    ]

    print("\nPosition DV01s (Dollar DV01 per 1bp):")
    for bond, notional_mm in zip(bonds, notionals):
        ytm = bond.price_to_yield(100.0, settlement_date)
        dv01_per_100 = bond.DV01(ytm, settlement_date)  # $ price change per 1bp per $100 face
        pos_dv01 = dv01_per_100 * (notional_mm * 1_000_000) / 100.0
        print(f"{bond.ticker:>4s}: Notional = {notional_mm:>8,.0f} MM | Position DV01 = ${pos_dv01:>12,.2f} per bp")
    print()

    total_dv01 = portfolio_dv01(bonds, notionals, settlement_date)
    print(f"Portfolio DV01: ${total_dv01:,.2f} per bp")

    hedge_mm = hedge_notional(total_dv01, bond10, settlement_date)
    print(f"10Y Hedge Notional: {hedge_mm:,.2f} MM")

    # Hedged portfolio = same 6 bonds, but 10Y notional adjusted
    bonds_hedged = bonds[:]                 
    notionals_hedged = notionals[:]         
    notionals_hedged[4] += hedge_mm         

    # ---------------- Scenario A ----------------
    shocks_A = [100, 100, 100, 100, 100, 100]  # bp

    pnl_A_unhedged = scenario_pnl(bonds, notionals, shocks_A, settlement_date)
    pnl_A_hedged   = scenario_pnl(bonds_hedged, notionals_hedged, shocks_A, settlement_date)

    print(f"Scenario A Unhedged PnL: ${pnl_A_unhedged:,.2f}")
    print(f"Scenario A Hedged PnL:   ${pnl_A_hedged:,.2f}")

    # ---------------- Scenario B ----------------
    shocks_B = [-10, -5, 0, 5, 10, 20]  # bp

    pnl_B_unhedged = scenario_pnl(bonds, notionals, shocks_B, settlement_date)
    pnl_B_hedged   = scenario_pnl(bonds_hedged, notionals_hedged, shocks_B, settlement_date)

    print(f"Scenario B Unhedged PnL: ${pnl_B_unhedged:,.2f}")
    print(f"Scenario B Hedged PnL:   ${pnl_B_hedged:,.2f}")

    # ---------------- Scenario C ----------------
    shocks_C = [-25, 0, 25, 25, 0, -25]  # bp

    pnl_C_unhedged = scenario_pnl(bonds, notionals, shocks_C, settlement_date)
    pnl_C_hedged   = scenario_pnl(bonds_hedged, notionals_hedged, shocks_C, settlement_date)

    print(f"Scenario C Unhedged PnL: ${pnl_C_unhedged:,.2f}")
    print(f"Scenario C Hedged PnL:   ${pnl_C_hedged:,.2f}")

    print(
    "\nQuestion 7 – Explanation\n\n"
    "The portfolio DV01 was calculated by summing the dollar DV01 of each bond position. "
    "A hedge was constructed using only the 10Y Treasury such that the total portfolio DV01 was neutralized.\n\n"
    "Scenario A (parallel +100 bp shift):\n"
    "The hedge works almost perfectly and the hedged PnL is near zero, because DV01 hedging is designed "
    "to protect against parallel yield curve shifts.\n\n"
    "Scenario B (non-parallel shift):\n"
    "The hedge does not fully eliminate PnL because yield changes differ by maturity. "
    "A single 10Y DV01 hedge cannot capture curve risk across multiple tenors.\n\n"
    "Scenario C (10Y unchanged):\n"
    "The hedge provides no protection since the 10Y yield does not move. "
    "The hedged and unhedged PnL are identical.\n\n"
    "Conclusion:\n"
    "DV01 hedging using a single maturity is effective for parallel shifts but fails under "
    "non-parallel yield curve movements.")


if __name__ == "__main__":
    main()