import datetime as dt
from calendar import monthrange
from pandas.tseries.holiday import USFederalHolidayCalendar
from fixed_income.dates.adjust import next_business_day
from .constants import MONTH_MAPPING

"""
Table of Contents:

1. us_fed_holiday_dates
2. us_sifma_holiday_dates
3. get_fomc_meeting_dates
4. get_fed_funds_future_settlement_date

"""

# Decision dates (2nd day of each meeting)
_FOMC_DECISION_DATES = {
    2025: [
        dt.date(2025, 1, 29),
        dt.date(2025, 3, 19),
        dt.date(2025, 5, 7),
        dt.date(2025, 6, 18),
        dt.date(2025, 7, 30),
        dt.date(2025, 9, 17),
        dt.date(2025, 10, 29),
        dt.date(2025, 12, 11),
    ],
    2026: [
        dt.date(2026, 1, 28),
        dt.date(2026, 3, 18),
        dt.date(2026, 4, 29),
        dt.date(2026, 6, 17),
        dt.date(2026, 7, 29),
        dt.date(2026, 9, 16),
        dt.date(2026, 10, 28),
        dt.date(2026, 12, 9),
    ],
    2027: [
        dt.date(2027, 1, 27),
        dt.date(2027, 3, 17),
        dt.date(2027, 4, 28),
        dt.date(2027, 6, 9),
        dt.date(2027, 7, 28),
        dt.date(2027, 9, 15),
        dt.date(2027, 10, 27),
        dt.date(2027, 12, 8),
    ],
}


def us_sifma_holiday_dates(year: int) -> set:
    """
    Return a set of US SIFMA holiday dates for the given year.

    Args:
        year (int): The year for which to get the holiday dates.

    Returns:        
        set: A set of datetime.date objects representing the holiday dates.
    """

    return {}

def get_fomc_meeting_dates(year: int) -> list[dt.date]:
    """
    Return a list of FOMC meeting (decision) dates for the given year.

    Note: These are the rate decision days (the 2nd day of each meeting).

    Args:
        year (int): Year (e.g., 2025)

    Returns:
        list[datetime.date]: FOMC decision dates for that year.
    """
    return _FOMC_DECISION_DATES.get(year, [])

def get_fed_funds_future_settlement_date(ticker) -> dt.date:
    """
    Get the settlement date for a Fed Funds Future based on its ticker and transaction date.

    Args:
        ticker (str): The ticker symbol of the Fed Funds Future (e.g., 'FFH3').

    Returns:
        datetime.date: The settlement date of the Fed Funds Future.
    """
    month_code = ticker[2]
    
    if month_code not in MONTH_MAPPING:
        raise ValueError(f"Invalid month code in ticker: {month_code}")
    
    year_code = ticker[3]
    month = MONTH_MAPPING[month_code]
    year = 2020+int(year_code)  # Assuming year is in the format '3' for 2023
    end_date = dt.date(year, month, monthrange(year, month)[1])
    
    settlement_date = next_business_day(end_date)

    return settlement_date