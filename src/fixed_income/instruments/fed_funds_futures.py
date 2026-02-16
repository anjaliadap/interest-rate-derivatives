from fixed_income.dates.constants import MONTH_MAPPING
from fixed_income.dates.utils import month_start, month_end, split_month_by_date
import re
import numpy as np
import datetime as dt
from scipy.linalg import lstsq

def parse_bbg_ff_ticker(ticker: str) -> tuple[int, int]:
    """
    Works for:
      FFV6
      FFV6 Comdty
      FFV6Comdty
    """

    # Remove whitespace entirely so all formats become comparable
    t = str(ticker).strip()

    # Match the first 4 characters: FF + month code + year digit
    m = re.match(r"^FF([FGHJKMNQUVXZ])(\d)", t)
    if not m:
        raise ValueError(f"Unrecognized Fed Funds ticker: {ticker}")

    month_code = m.group(1)
    year_digit = int(m.group(2))

    month = MONTH_MAPPING[month_code]
    year = 2020 + year_digit

    return year, month

class FedFundsFutures:
    
    def __init__(self, ticker: str, price: float):
        self.ticker = ticker

        self.year, self.month = parse_bbg_ff_ticker(ticker) # parse ticker -> (year, month)

        # contract month start/end dates
        # for FFF, each contract is tied to one calendar month
        self.start_date = month_start(self.year, self.month)
        self.end_date = month_end(self.year, self.month)

        # price + implied rate
        self.price = float(price)
        self.implied_rate = round(100.0 - self.price, 3)

    @staticmethod
    def extract_fomc_expectations_bootstrap(contracts, effr, fomc_dates, min_after_weight=0.25):

        contracts = sorted(contracts, key=lambda c: c.start_date)
        fomc_dates = sorted(fomc_dates)

        current_rate = float(effr)
        meeting_expectations = {}

        for i, c in enumerate(contracts):
            Rm = float(c.implied_rate)
            meetings_in_month = [d for d in fomc_dates if c.start_date <= d <= c.end_date]

            # edge cases - no meetings or multiple meetings in a month
            if len(meetings_in_month) == 0:
                # A no-meeting month is basically a direct read of the current regime rate
                current_rate = Rm
                continue

            if len(meetings_in_month) > 1:
                raise ValueError(f"Multiple FOMC meetings in {c.start_date:%Y-%m}: {meetings_in_month}")

            mdate = meetings_in_month[0]
            w_before, w_after = split_month_by_date(c.start_date, c.end_date, mdate)

            if w_after >= min_after_weight:
                # Safe to solve from this month
                new_rate = (Rm - w_before * current_rate) / w_after
                meeting_expectations[mdate] = round(new_rate, 3)
                current_rate = new_rate
            else:
                # Meeting is too late -> infer post-meeting rate from next no-meeting month
                next_anchor = None
                for j in range(i + 1, len(contracts)):
                    c2 = contracts[j]
                    meetings_next = [d for d in fomc_dates if c2.start_date <= d <= c2.end_date]
                    if len(meetings_next) == 0:
                        next_anchor = float(c2.implied_rate)
                        break

                if next_anchor is None:
                    raise ValueError(
                        f"Meeting {mdate} is late in month and no later no-meeting month exists in the strip."
                    )

                meeting_expectations[mdate] = round(next_anchor, 3)
                current_rate = next_anchor

        return meeting_expectations

    @staticmethod
    def extract_fomc_expectations_lstsq(contracts, effr, fomc_dates):

        def _overlap_days(a0: dt.date, a1: dt.date, b0: dt.date, b1: dt.date) -> int:
            """Inclusive overlap days between [a0,a1] and [b0,b1]."""
            start = max(a0, b0)
            end = min(a1, b1)
            if end < start:
                return 0
            return (end - start).days + 1
        
        contracts = sorted(contracts, key=lambda c: c.start_date)
        meetings = sorted(fomc_dates)
                
        if len(meetings) == 0:
            raise ValueError("Need at least one FOMC date to run least squares method.")

        K = len(meetings)
        
        # Unknown regimes:
        # regime k corresponds to [meeting_k, day_before_meeting_(k+1)] (last goes far future)
        unknown_starts = meetings
        unknown_ends = [meetings[i+1] - dt.timedelta(days=1) for i in range(K-1)] + [dt.date(2100, 1, 1)]

        A_rows = []
        b_vals = []

        for c in contracts:
            m0, m1 = c.start_date, c.end_date
            total_days = (m1 - m0).days + 1

            # Known pre-first-meeting part (rate = effr)
            pre_end = meetings[0] - dt.timedelta(days=1)
            pre_days = _overlap_days(m0, m1, dt.date(1900, 1, 1), pre_end)
            w0 = pre_days / total_days

            # Unknown weights
            wk = []
            for s, e in zip(unknown_starts, unknown_ends):
                d = _overlap_days(m0, m1, s, e)
                wk.append(d / total_days)

            # b = Rm - w0*effr
            A_rows.append(wk)
            b_vals.append(float(c.implied_rate) - w0 * float(effr))

        A = np.array(A_rows, dtype=float)
        b = np.array(b_vals, dtype=float)

        # Solve A x ≈ b
        x, residuals, rank, svals = lstsq(A, b)

        return {meetings[i]: round(float(x[i]), 3) for i in range(K)}