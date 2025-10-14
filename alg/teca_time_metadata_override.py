"""
TECA algorithm to override time axis metadata of an input dataset.

This module provides the teca_time_metadata_override class, which allows users to
replace the time axis metadata (time values, calendar, and units) in a dataset
without modifying the actual data values.
"""

import numpy as np


class teca_time_metadata_override(teca_python_algorithm):
    """
    A TECA algorithm to override the time axis metadata of an input dataset.

    This algorithm allows you to replace the time values, calendar, and units
    attributes in the metadata of a dataset. The actual data arrays are passed
    through unchanged; only the metadata describing the time axis is modified.

    Attributes:
        time_values (numpy.ndarray): The time values to use in the output metadata.
        calendar (str): The calendar attribute for the output metadata.
        units (str): The units attribute for the output metadata.

    Methods:
        set_time_values(time_values): Set the time values to use.
        set_calendar(calendar): Set the calendar attribute.
        set_units(units): Set the units attribute.
        get_time_values(): Get the currently configured time values.
        get_calendar(): Get the currently configured calendar.
        get_units(): Get the currently configured units.
        report(port, md_in): TECA report phase - updates metadata.
        execute(port, data_in, req): TECA execute phase - passes data through.
    """

    def __init__(self):
        """Initialize the time metadata override algorithm."""
        super(teca_time_metadata_override, self).__init__()
        self.time_values = None
        self.calendar = None
        self.units = None

    def set_time_values(self, time_values):
        """
        Set the time values to use in the output metadata.

        Args:
            time_values: Array-like object containing the time values. Should be
                         convertible to a numpy array.

        Raises:
            TypeError: If time_values is None or cannot be converted to array.
            ValueError: If time_values is empty.
        """
        if time_values is None:
            raise TypeError("time_values cannot be None")

        # Convert to numpy array if not already
        time_values = np.asarray(time_values)

        if time_values.size == 0:
            raise ValueError("time_values cannot be empty")

        self.time_values = time_values

    def set_calendar(self, calendar):
        """
        Set the calendar attribute for the output metadata.

        Args:
            calendar: String specifying the calendar system. Common CF-compliant
                     calendars include: 'standard', 'gregorian', 'proleptic_gregorian',
                     'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day'.

        Raises:
            TypeError: If calendar is None.
            ValueError: If calendar is not a non-empty string.
        """
        if calendar is None:
            raise TypeError("calendar cannot be None")

        if not isinstance(calendar, str):
            raise TypeError(f"calendar must be a string, got {type(calendar).__name__}")

        if not calendar.strip():
            raise ValueError("calendar cannot be an empty string")

        self.calendar = calendar

    def set_units(self, units):
        """
        Set the units attribute for the output metadata.

        Args:
            units: String specifying the time units (e.g., 'days since 2000-01-01').

        Raises:
            TypeError: If units is None.
            ValueError: If units is not a non-empty string.
        """
        if units is None:
            raise TypeError("units cannot be None")

        if not isinstance(units, str):
            raise TypeError(f"units must be a string, got {type(units).__name__}")

        if not units.strip():
            raise ValueError("units cannot be an empty string")

        self.units = units

    def get_time_values(self):
        """
        Get the currently configured time values.

        Returns:
            numpy.ndarray or None: The time values array, or None if not yet set.
        """
        return self.time_values

    def get_calendar(self):
        """
        Get the currently configured calendar.

        Returns:
            str or None: The calendar string, or None if not yet set.
        """
        return self.calendar

    def get_units(self):
        """
        Get the currently configured time units.

        Returns:
            str or None: The units string, or None if not yet set.
        """
        return self.units

    def report(self, port, md_in):
        """
        TECA report phase - updates the time axis metadata.

        This method modifies the metadata to reflect the new time values,
        calendar, and units that were set via the setter methods.

        Args:
            port: The input port number.
            md_in: List of teca_metadata objects from upstream algorithms.

        Returns:
            teca_metadata: Modified metadata with updated time axis information.

        Raises:
            RuntimeError: If time_values, calendar, or units have not been set.
            KeyError: If required metadata keys are missing.
        """
        # Validate that all required properties have been set
        if self.time_values is None:
            raise RuntimeError(
                "time_values must be set before calling report(). "
                "Use set_time_values() to provide time values."
            )

        if self.calendar is None:
            raise RuntimeError(
                "calendar must be set before calling report(). "
                "Use set_calendar() to provide a calendar string."
            )

        if self.units is None:
            raise RuntimeError(
                "units must be set before calling report(). "
                "Use set_units() to provide a units string."
            )

        # Create output metadata from input teca_metadata object
        # Pattern adopted from teca_temporal_reduction.py
        md_out = md_in[0]

        try:
            atts = md_out['attributes']
            coords = md_out['coordinates']
        except KeyError as e:
            raise KeyError(
                f"Required metadata key {e} is missing from input metadata. "
                "Input must contain 'attributes' and 'coordinates' keys."
            )

        # Update time coordinate values
        coords['t'] = self.time_values

        try:
            t_var = coords['t_variable']
        except KeyError:
            raise KeyError(
                "Metadata is missing 't_variable' in coordinates. "
                "Cannot determine which variable contains time attributes."
            )

        try:
            t_atts = atts[t_var]
        except KeyError:
            raise KeyError(
                f"Time variable '{t_var}' not found in attributes. "
                "Cannot update time metadata."
            )

        # Update coordinates with new time values
        md_out['coordinates'] = coords

        # Update time attributes with new calendar and units
        t_atts['calendar'] = self.calendar
        t_atts['units'] = self.units
        atts[t_var] = t_atts

        # Update attributes in output metadata
        md_out['attributes'] = atts

        return md_out

    def execute(self, port, data_in, req):
        """
        TECA execute phase - passes data through unchanged.

        This algorithm only modifies metadata, not data, so the execute phase
        simply returns the input data unchanged.

        Args:
            port: The input port number.
            data_in: List of input data objects.
            req: The request object.

        Returns:
            The input data unchanged.
        """
        return data_in[port]