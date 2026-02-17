from calendar import monthrange
import numpy as np
import datetime as dt

"""
Table of Contents:

1. find_index
2. find_index_for_dates
3. arrange_dates_chronologically
4. arrange_terms_chronologically

"""

def month_start(year: int, month: int) -> dt.date:
    """Return the first calendar day of the given month."""
    return dt.date(year, month, 1)

def month_end(year: int, month: int) -> dt.date:
    """Return the last calendar day of the given month."""
    last_day = monthrange(year, month)[1]
    return dt.date(year, month, last_day)

def find_index(list_of_values, value) -> int:
    """Find the index of the largest element in list_of_values less than or equal to value.

    Args:
        list_of_values (list): A sorted list of values.
        value (float): The value to compare against.

    Returns:
        int: The index of the largest element less than or equal to value.
    """
    target_index = np.max(np.flatnonzero( np.array(list_of_values) <= value ))

    # target_value = list_of_values[int(target_value)]
    return target_index

def find_index_for_dates(list_of_dates, target_date) -> int:
    """Find the index of the largest element in list_of_values less than or equal to value.


    Args:
        list_of_dates (list): A sorted list of dates.
        value (float): The value to compare against.

    Returns:
        int: The index of the largest date less than or equal to value.
    """
    list_of_datetimes = [dt.datetime(this_date.year, this_date.month, this_date.day) for this_date in list_of_dates]
    list_of_values = [this_datetime.timestamp()/86400.0 for this_datetime in list_of_datetimes] # number of days since 1970(?)
    value = dt.datetime(target_date.year, target_date.month, target_date.day).timestamp()/86400.0
    target_index = np.max(np.flatnonzero( np.array(list_of_values) < value ))

    #target_value = list_of_values[int(target_value)]
    return target_index

def arrange_dates_chronologically(date_list: list) -> list:
    """Arrange a list of dates in chronological order.

    Args:
        date_list (list): A list of datetime.date objects.

    Returns:
        list: A list of datetime.date objects arranged in chronological order.
    """
    return sorted(date_list)

def arrange_terms_chronologically(term_list: list) -> list:
    """Arrange a list of terms (e.g., '5Y', '6M') in chronological order.

    Args:
        term_list (list): A list of term strings.

    Returns:
        list: A list of term strings arranged in chronological order.
    """
    def term_to_months(term: str) -> int:
        num = int(term[:-1])
        unit = term[-1]
        if unit == 'Y':
            return num * 12
        elif unit == 'M':
            return num
        else:
            raise ValueError(f"Invalid term unit: {unit}")

    return sorted(term_list, key=term_to_months)

def split_month_by_date(month_begin: dt.date, month_end: dt.date, split_date: dt.date) -> tuple[float, float]:
    """
    Split a month into two fractions based on a date inside the month.

    Returns:
      frac_before: fraction of days strictly before split_date
      frac_after:  fraction of days split_date or later

    Fractions sum to 1.0.
    """

    # Basic validation
    if month_begin > month_end:
        raise ValueError("month_begin must be <= month_end")

    if split_date < month_begin or split_date > month_end:
        raise ValueError("split_date must be inside the month")

    total_days = (month_end - month_begin).days + 1

    days_before = (split_date - month_begin).days
    days_after = (month_end - split_date).days + 1

    frac_before = days_before / total_days
    frac_after = days_after / total_days

    return frac_before, frac_after
