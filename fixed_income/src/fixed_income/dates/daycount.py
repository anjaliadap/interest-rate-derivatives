import datetime

"""
Table of contents:

1. year_fraction

"""

def year_fraction(start: datetime, end: datetime, day_count_convention = "ACT/360") -> float:
        """
        Calculate year fraction using ACT/360.

        Args:
            start (datetime): Start date.
            end (datetime): End date.
            day_count_convention (str): Day count convention to use. Default is "ACT/360".
        
        Returns:
            float: Year fraction between start and end dates.
        """

        days = (end - start).days
        if day_count_convention == "ACT/360":
            return days / 360.0
        
        return days / 365.25