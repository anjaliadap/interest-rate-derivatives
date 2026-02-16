import datetime as dt
import math

class FFERCurve:
    """
    Minimal FFER front-end curve 

    - Stores curve_date, effr, and market-expected post-meeting rates.
    - Flat-forward: the daily rate is piecewise constant between FOMC dates.
    - Builds discount factors day-by-day using:
          DF(d) = DF(d-1) / (1 + r_d/360)
    """

    def __init__(self, curve_date: dt.date, effr: float,
                 meeting_rates: dict[dt.date, float],
                 meeting_calendar: list[dt.date]):
        self.curve_date = curve_date
        self.effr = float(effr)

        # meeting_rates: {meeting_date: expected post-meeting rate (%)} for solved meetings
        self.meeting_rates = dict(sorted(meeting_rates.items()))

        # meeting_calendar: full list of meeting dates (needs to include at least the next unsolved one)
        self.meeting_calendar = sorted(meeting_calendar)

        if len(self.meeting_rates) == 0:
            raise ValueError("meeting_rates is empty (need solved FOMC expectations).")
        if len(self.meeting_calendar) == 0:
            raise ValueError("meeting_calendar is empty (need FOMC dates).")

    def max_supported_date(self) -> dt.date:
        """
        Returns the next succeeding FOMC date that is NOT solved for.
        here March 2027
        """
        last_solved = max(self.meeting_rates.keys())
        future = [d for d in self.meeting_calendar if d > last_solved]
        if not future:
            raise ValueError("No meeting after last solved meeting; extend meeting_calendar.")
        return future[0]

    def rate_for_date(self, d: dt.date) -> float:
        """
        Daily FFER (%), flat between meetings:
        - Before the first solved meeting date: use effr
        - From a solved meeting date onward: use that meeting's expected post-meeting rate
        """
        solved = [m for m in self.meeting_rates.keys() if m <= d]
        if not solved:
            return self.effr
        return self.meeting_rates[max(solved)]

    def discount_factor(self, target_date: dt.date) -> float:
        """
        Return DF(curve_date -> target_date) by iterating day-by-day.

        DF(curve_date) = 1.0
        For each next day d:
            DF(d) = DF(d-1) / (1 + r_d/360)
        """
        if target_date < self.curve_date:
            raise ValueError("target_date must be >= curve_date")

        cutoff = self.max_supported_date()
        if target_date > cutoff:
            raise ValueError(f"target_date beyond cutoff {cutoff} (next unsolved FOMC date).")

        df = 1.0
        d = self.curve_date

        while d < target_date:
            d = d + dt.timedelta(days=1)

            r = self.rate_for_date(d)  # percent -like 3.585
            df = df / (1.0 + (r / 100.0) / 360.0)

        return df

    def yearfrac_act360(self, d1, d2):
        return (d2 - d1).days / 360.0

    def implied_simple_forward_rate(self, d1, d2):
        df1 = self.discount_factor(d1)
        df2 = self.discount_factor(d2)
        tau = self.yearfrac_act360(d1, d2)
        return (df1 / df2 - 1.0) / tau
    
    def discount_factor_fwd(self, maturity_date: dt.date, forward_curve_date: dt.date | None = None) -> float:
        """
        Forward discount factor (forward ZCB price):
            DF_fwd(T; f) = DF(T) / DF(f)

        If forward_curve_date is None, use curve_date (spot DF).
        """
        if forward_curve_date is None:
            forward_curve_date = self.curve_date

        if maturity_date < forward_curve_date:
            raise ValueError("maturity_date must be >= forward_curve_date")

        return self.discount_factor(maturity_date) / self.discount_factor(forward_curve_date)

    def daily_forward_rate(self, this_date: dt.date, forward_curve_date: dt.date | None = None) -> float:
        """
        Daily forward rate for a single day starting at this_date (ACT/360),
        computed from forward discount factors:

            1 + f/360 = DF_fwd(this_date; fwd_date) / DF_fwd(this_date+1; fwd_date)

        Returns a decimal (0.035 = 3.5%).
        """
        if forward_curve_date is None:
            forward_curve_date = self.curve_date

        d0 = this_date
        d1 = this_date + dt.timedelta(days=1)

        df0 = self.discount_factor_fwd(d0, forward_curve_date)
        df1 = self.discount_factor_fwd(d1, forward_curve_date)

        # df1 should be <= df0; ratio > 1 implies positive rate
        return 360.0 * (df0 / df1 - 1.0)

    def zcb_yield(self, maturity_date: dt.date, forward_curve_date: dt.date | None = None) -> float:
        """
        Zero-coupon bond yield implied by discount factors (continuous comp, ACT/360):

            y = -ln(DF_fwd(T; f)) / tau

        Returns a decimal (0.035 = 3.5%).
        """
        if forward_curve_date is None:
            forward_curve_date = self.curve_date

        if maturity_date < forward_curve_date:
            raise ValueError("maturity_date must be >= forward_curve_date")

        tau = (maturity_date - forward_curve_date).days / 360.0
        if tau == 0:
            raise ValueError("maturity_date equals forward_curve_date (tau=0)")

        df_fwd = self.discount_factor_fwd(maturity_date, forward_curve_date)
        
        return -math.log(df_fwd) / tau

    def money_market_rate(self, start_date: dt.date, end_date: dt.date, forward_curve_date: dt.date | None = None) -> float:
        """
        Money market term rate over [start_date, end_date] (simple, ACT/360),
        using forward discount factors:

            DF_fwd(end; f) / DF_fwd(start; f) = 1 / (1 + r * tau)
        =>  r = (DF_fwd(start)/DF_fwd(end) - 1) / tau

        Returns a decimal (0.035 = 3.5%).
        """
        if forward_curve_date is None:
            forward_curve_date = self.curve_date

        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")

        tau = (end_date - start_date).days / 360.0
        if tau == 0:
            raise ValueError("start_date equals end_date (tau=0)")

        df_start = self.discount_factor_fwd(start_date, forward_curve_date)
        df_end = self.discount_factor_fwd(end_date, forward_curve_date)

        return (df_start / df_end - 1.0) / tau