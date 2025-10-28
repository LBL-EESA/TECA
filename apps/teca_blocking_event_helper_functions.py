# Helper functions to sync daily thresholds with input data time steps:
import sys
import teca
import numpy as np
from datetime import datetime


def report_error_and_exit(message, rank, exit_code=-1):
    """
    Report an error message and exit the program.

    Parameters
    ----------
    message : str
        Error message to display
    rank : int
        MPI rank (only rank 0 will print the message)
    exit_code : int, optional
        Exit code to use (default: -1)
    """
    if rank == 0:
        sys.stderr.write(f'ERROR: {message}\n')
    sys.exit(exit_code)


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

def calculate_elapsed_seconds(input_md, first_step, last_step):
    """
    Calculate elapsed seconds since time step 0 for the processed time range.

    Parameters
    ----------
    input_md : teca_metadata
        Input metadata containing time coordinate information
    first_step : int
        First time step to process
    last_step : int
        Last time step to process (inclusive)

    Returns
    -------
    elapsed_seconds : numpy.ndarray
        Array of elapsed seconds since time step 0 for each processed time step
    """
    input_coords = input_md['coordinates']
    t = input_coords['t']

    # Get calendar and units from metadata
    time_atts = input_md['attributes']['time']
    calendar = time_atts['calendar']
    units = time_atts['units']

    # Convert time step 0 to datetime
    t0 = t[0]
    year0, month0, day0, hour0, minute0, second0 = teca.calendar_util_date(t0, units, calendar)
    dt0 = datetime(year0, month0, day0, hour0, minute0, int(second0))

    # Calculate elapsed seconds for each time step in the range
    elapsed_seconds = []
    for i in range(first_step, last_step + 1):
        year, month, day, hour, minute, second = teca.calendar_util_date(t[i], units, calendar)
        curr_dt = datetime(year, month, day, hour, minute, int(second))
        elapsed = (curr_dt - dt0).total_seconds()
        elapsed_seconds.append(elapsed)

    return np.array(elapsed_seconds)


def print_metadata_times(label, input_md):
    """
    Print the time values from the input metadata for debugging purposes.

    Parameters
    ----------
    input_md : teca_metadata
        Input metadata containing time coordinate information
    """
    input_coords = input_md['coordinates']
    t = input_coords['t']

    # Get calendar and units from metadata
    time_atts = input_md['attributes']['time']
    calendar = time_atts['calendar']
    units = time_atts['units']

    print(f"{label} metadata time values (calendar: {calendar}, units: {units}):")
    for i, time_value in enumerate(t):
        year, month, day, hour, minute, second = teca.calendar_util_date(time_value, units, calendar)
        print(f"Time step {i}: time value {time_value} -> {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{int(second):02d}")