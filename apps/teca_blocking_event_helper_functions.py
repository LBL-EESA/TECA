# Helper functions to sync daily thresholds with input data time steps:
import teca

def day_of_year(month: int, day: int) -> int:
# (written by CBorg AI at LBNL)
    """
    Return the day of the year (1-365) for the given month and day.

    Input: month (1-12), day (1-31)
    Output: day of year (1-365)
    """
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if not (1 <= month <= 12):
        raise ValueError("Month must be 1..12")
    if not (1 <= day <= month_lengths[month-1]):
        raise ValueError("Invalid day for given month")
    return sum(month_lengths[:month-1]) + day

def day_in_year(t, units, calendar):
    """
    Convert a time value from a NetCDF file read by TECA to the corresponding
    day of the year (1-365).

    NOTE: This function maps Feb 29th to March 1st to avoid leap years.

    Input: t (time value), units (time units), calendar (calendar type)
    Output: day of year (1-365)
    """
    year, month, day, _, _, _ = teca.calendar_util_date(t, units, calendar)
    if month == 2 and day == 29:
        month, day = 3, 1 # No leap years, map Feb 29th to March 1st
    return day_of_year(month, day)

def generate_day_index_list(md):
    """
    Generate a list of day indices (0-364) for the thresholds data based on
    the time. This list is used to sync the thresholds file with input file
    time steps.

    Input: md (metadata dictionary from TECA for input file
    Output: list of day indices (0-364) for each time step in input file
    """
    # Generate an array of day indices for the thresholds data. Iterate over
    # time values in the metaddata and convert to day of year. This is the
    # index into the thrsholds data set. (Note: Thresholds data does not have
    # leap days. Feb 29th is mapped to March 1st.)
    time_atts = md['attributes']['time']
    calendar = time_atts['calendar']
    units = time_atts['units']
    times = md['coordinates']['t']
    index_list = [day_in_year(t, units, calendar) - 1 for t in times]
    for pos, index in enumerate(index_list):
        if index < 0 or index >= 365:
            raise ValueError(f'Invalid day index {index} for position {pos} '
                             f'at time {times[pos]} in thresholds data.')
    return index_list

def validate_thresholds_file(thresholds_md, rank=0):
    """
    Validate that the thresholds file contains exactly one year of daily data.

    Parameters
    ----------
    thresholds_md : dict
        Metadata dictionary from TECA for the thresholds file
    rank : int, optional
        MPI rank for conditional error reporting (default: 0)

    Returns
    -------
    bool
        True if the thresholds file is valid, False otherwise
    """
    import sys

    # Map out the time axis of the thresholds data. It must contain exactly
    # one year of daily data (365 time steps). If not, return an error.
    index_list = generate_day_index_list(thresholds_md)
    if index_list != list(range(365)):
        if rank == 0:
            print(index_list)
            sys.stderr.write('ERROR: The thresholds data set must contain '
                             'exactly one year of daily data\n')
        return False
    else:
        return True
