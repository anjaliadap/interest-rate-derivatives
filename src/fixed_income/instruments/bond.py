import datetime as dt
from fixed_income.dates.adjust import add_months
from fixed_income.curves.zero_curve import ZeroCurve

class FixedRateBond:
    
    """
    This is a class representing a fixed rate bond instrument.
    """

    def __init__(self
                 , ticker: str
                 , coupon_rate: float
                 , maturity_date: dt.date
                 , issue_date: dt.date
                 , dated_date: dt.date = None
                 , yield_to_maturity: float = None
                 , face_value: float = 100.00
                 , coupon_frequency: int = 2
                 ):
        
        self.ticker = ticker
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_date = maturity_date
        self.issue_date = issue_date
        self.coupon_frequency = coupon_frequency
        self.yield_to_maturity = yield_to_maturity
        
        # Dated date is the date from which interest starts accruing.
        # If not provided, it defaults to the issue date.
        self.dated_date = issue_date if dated_date is None else dated_date
        self.coupon_dates = self._get_coupon_dates(self.dated_date, self.maturity_date, self.coupon_frequency)
        self.coupon_periods = self._get_coupon_periods(self.coupon_dates)
        
    def _get_coupon_dates(self, dated_date: dt.date, maturity_date: dt.date, coupon_frequency):
        """
        Generate coupon payment dates between issue_date and maturity_date.

        Args: 
            dated_date (dt.date): The dated date of the bond 
            maturity_date (dt.date): The maturity date of the bond
        
        Returns: 
            list: A list of coupon payment dates.
        """
        step = 12 // coupon_frequency
        coupon_dates=[]
        coupon_date=dated_date

        while True: 
            coupon_date=add_months(coupon_date, step)

            if coupon_date >= maturity_date:
                break   
            
            coupon_dates.append(coupon_date)

        # Add maturity date as final coupon date
        if not coupon_dates or coupon_dates[-1] != maturity_date:
            coupon_dates.append(maturity_date)

        return coupon_dates
    
    def _get_coupon_periods(self, coupon_dates):
        """
        This function gets coupon periods for the bond.

        Args:
            coupon_dates (list): This is the list of the coupon dates.
        
        Returns:
            coupon_periods (list of tuples): This is the list of tuples with a start_date and end_date for each coupon_period.
        """
        coupon_periods=[(self.dated_date, coupon_dates[0])]
        for i in range(len(coupon_dates)-1):
            period = (coupon_dates[i], coupon_dates[i+1])
            coupon_periods.append(period)

        return coupon_periods


    def yield_to_pv(self, settlement_date: dt.date, yield_to_maturity: float) -> float:
        """
        Given yield and settlement date of the bond, calculate its present value.
        This includes the accrued interest. 

        Args:
            settlement_date (dt.date): The settlement date for the bond.
            yield_to_maturity (float): The yield to maturity as a decimal.
        
        Returns:
            float: The present value of the bond.
        """

        # No future cashflows if settled on or after maturity
        if settlement_date >= self.maturity_date:
            return 0.0

        pv = 0.0
        period_number = 0
        tau = None # The year-fraction
        freq = self.coupon_frequency
        coupon_payment = (self.coupon_rate / freq) * self.face_value

        for period_start, period_end in self.coupon_periods:

            # Skip past coupons
            if period_end <= settlement_date:
                continue

            # Fraction of current coupon period remaining at settlement
            if tau is None:
                tau = (period_end - settlement_date).days / (period_end - period_start).days

            # Discount exponent in coupon periods
            exponent = period_number + tau
            # Discount factor with compounding
            df = (1.0 + yield_to_maturity / freq) ** exponent

            # Coupon PV
            pv += coupon_payment / df

            # Principal PV at maturity
            if period_end == self.maturity_date:
                pv += self.face_value / df

            period_number += 1

        return pv
    
    def accrued_interest(self, settlement_date):
        """
        This function gets the accrued interest in the period that contains the settlement date. 

        Args: 
            settlement_date (dt.date): This is the settlement date of the bond.

        Returns:
            accrued_interest (float): This is the total accrued interest from the last coupon date to the settlement date.
        """
        if settlement_date <= self.dated_date:
            return 0.0

        last_coupon_date = None
        next_coupon_date = None

        # Get the start and end dates of the period to which the settlement date belongs 
        for period_start, period_end in self.coupon_periods:
            if period_end > settlement_date:
                next_coupon_date = period_end
                last_coupon_date = period_start
                break

        if last_coupon_date is None or next_coupon_date is None:
            return 0.0
        
        # Days between the last coupon date and the settlement date 
        accrued_days = (settlement_date - last_coupon_date).days
        # Total number of days in this period 
        total_days = (next_coupon_date - last_coupon_date).days
        # Calculate the accrued interest
        accrued_interest = self.face_value * (self.coupon_rate / self.coupon_frequency) * (accrued_days / total_days)

        return accrued_interest    

    def yield_to_quoted_price(self, yield_rate, settlement_date):
        """
        Calculate the quoted price of the USTreasuryBond, which excludes accrued interest.

        Parameters:
            yield_rate (float): The annual yield rate (as a decimal).
            settlement_date (datetime.date): The date on which the bond is being priced.

        Returns:
            float: The quoted price of the bond, which excludes accrued interest.
        """
        # Get the present value of the bond - dirty price 
        pv = self.yield_to_pv(settlement_date=settlement_date, yield_to_maturity=yield_rate)
        # Get the accrued interest 
        accrued_interest = self.accrued_interest(settlement_date)
        
        # Clean price = Dirty price - Accrued interest
        return pv - accrued_interest
                    
    def price_to_yield_bisection(self, market_price, settlement_date, tol=1e-6, max_iter=100):
        """
        Calculate the yield to maturity of the bond given a market price and settlement date.
        This calculation makes use of the bisection method. 

        Parameters:
            market_price (float): The market price of the bond.
            settlement_date (datetime.date): The date on which the bond is being priced.
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.

        Returns:
            float: The yield to maturity (as a decimal).    
        """
        lower_bound = 0.0
        upper_bound = 1.0

        for _ in range(max_iter): 
            mid_yield = (lower_bound + upper_bound) / 2
            price = self.yield_to_quoted_price(mid_yield, settlement_date)

            if abs(price - market_price) < tol:
                return mid_yield
            if price < market_price:
                upper_bound = mid_yield
            else:
                lower_bound = mid_yield

        raise ValueError("Yield to maturity calculation did not converge")

    def price_to_yield_newton_raphson(self, market_price, settlement_date, initial_guess=0.05, tol=1e-6, max_iter=100):
        """
        Calculate the yield to maturity of the bond given a market price and settlement date.
        This calculation makes use of the Newton-Raphson method. 

        Parameters:
            market_price (float): The market price of the bond.
            settlement_date (datetime.date): The date on which the bond is being priced.
            initial_guess (float): The initial guess for the yield to maturity.
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.

        Returns:
            float: The yield to maturity (as a decimal).    
        """
        yield_rate = initial_guess

        for _ in range(max_iter):
            price = self.yield_to_quoted_price(yield_rate, settlement_date)

            # Numerical derivative (finite difference)
            h = 1e-5
            price_plus_h = self.yield_to_quoted_price(yield_rate + h, settlement_date)
            derivative = (price_plus_h - price) / h

            if derivative == 0:
                raise ValueError("Zero derivative encountered in Newton-Raphson method")

            # Update yield using Newton-Raphson formula
            yield_rate -= (price - market_price) / derivative

            if abs(price - market_price) < tol:
                return yield_rate

        raise ValueError("Yield to maturity calculation did not converge")

    def price_to_coupon(self, price, base_yield, settlement_date, tol=1e-6, max_iter=100): 
        """
        Calculate the coupon rate of the bond given a market price and settlement date.

        Parameters:
            market_price (float): The market price of the bond.
            settlement_date (datetime.date): The date on which the bond is being priced.
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.

        Returns:
            float: The coupon rate (as a decimal).
        """
        bond = FixedRateBond(
            ticker=self.ticker,
            coupon_rate=0.0,  # Initial guess for coupon rate
            maturity_date=self.maturity_date,
            issue_date=self.issue_date,
            dated_date=self.dated_date,
            face_value=self.face_value,
            coupon_frequency=self.coupon_frequency)


        lower_bound = 0.0
        upper_bound = 0.5

        for _ in range(max_iter):
            mid_coupon = (lower_bound + upper_bound) / 2
            bond.coupon_rate = mid_coupon
            mid_price = bond.yield_to_quoted_price(base_yield, settlement_date)

            if abs(price - mid_price) < tol:
                return mid_coupon
            if price < mid_price: # if price is less than mid_price, need to lower coupon
                upper_bound = mid_coupon
            else:
                lower_bound = mid_coupon

        raise ValueError("Coupon rate calculation did not converge")

    def pv_from_zero_curve(self, zero_curve: ZeroCurve, settlement_date): 
        """
        Calculate the price of the FixedRateBond using a zero-coupon bond yield curve.
        
        Args:
            zero_curve (ZeroCurve): An instance of ZeroCurve that provides zero rates and discount factors.
            settlement_date (datetime.date): The date on which the bond is being priced.
        
        Returns:
            float: The pv of the bond.
            IMPORTANT: pv is defined as the sum of the present values of all future cash flows (coupons that are paid do not contribute),
                    discounted using the zero-coupon yield curve.
            """
        pv = 0.0
        freq = self.coupon_frequency

        if settlement_date >= self.maturity_date:
            return 0.0

        coupon_payment = (self.coupon_rate / freq) * self.face_value

        for period_start, period_end in self.coupon_periods:
            if period_end <= settlement_date:
                continue

            tau = (period_end - settlement_date).days / 365.0
            df = zero_curve.get_discount_factor(tau)

            pv += coupon_payment * df

            if period_end == self.maturity_date:
                pv += self.face_value * df

        return pv

    def CPN01(self, yield_rate, settlement_date, delta_coupon=0.0001): 
        """
        Calculate the CPN01 (Coupon Value of 01) of the Bond.
        This is the change in price for a 1 basis point change in coupon rate.

        Args:
            yield_rate (float): The annual yield rate (as a decimal).
            settlement_date (datetime.date): The date on which the bond is being priced.
            delta_coupon (float): The change in coupon rate to calculate CPN01.

        Returns:
            float: The CPN01 of the bond.
        """
        bond_up = FixedRateBond(
            ticker=self.ticker,
            coupon_rate=self.coupon_rate + delta_coupon,
            maturity_date=self.maturity_date,
            issue_date=self.issue_date,
            dated_date=self.dated_date,
            face_value=self.face_value,
            coupon_frequency=self.coupon_frequency
        )
        price_up = bond_up.yield_to_quoted_price(yield_rate, settlement_date)

        bond_down = FixedRateBond(
            ticker=self.ticker,
            coupon_rate=self.coupon_rate - delta_coupon,
            maturity_date=self.maturity_date,
            issue_date=self.issue_date,
            dated_date=self.dated_date,
            face_value=self.face_value,
            coupon_frequency=self.coupon_frequency
        )
        price_down = bond_down.yield_to_quoted_price(yield_rate, settlement_date)

        return (price_up - price_down) / 2.0

    def DV01(self, yield_rate, settlement_date, delta_yield=0.0001):
        """
        DV01: Dollar value of a 1bp change in yield.
        By convention DV01 is positive.

        Args:
            yield_rate (float): The annual yield rate (as a decimal).
            settlement_date (datetime.date): The date on which the bond is being priced.
            delta_yield (float): The change in yield to calculate DV01.

        Returns:
            float: The DV01 of the bond.
        """

        price_up = self.yield_to_quoted_price(yield_rate + delta_yield, settlement_date)
        price_down = self.yield_to_quoted_price(yield_rate - delta_yield, settlement_date)

        return (price_down - price_up) / 2.0

    def convexity(self, yield_rate, settlement_date, delta_yield=0.0001):
        """
        Calculate the convexity of the USTreasuryBond.

        Args:
            yield_rate (float): The annual yield rate (as a decimal).
            settlement_date (datetime.date): The date on which the bond is being priced.
            delta_yield (float): The change in yield to calculate convexity.

        Returns:
            float: The convexity of the bond.
        """
        p_up = self.yield_to_quoted_price(yield_rate + delta_yield, settlement_date)
        p_dn = self.yield_to_quoted_price(yield_rate - delta_yield, settlement_date)
        p_0  = self.yield_to_quoted_price(yield_rate, settlement_date)

        d2p = (p_up + p_dn - 2.0 * p_0) / (delta_yield ** 2)

        return d2p / p_0