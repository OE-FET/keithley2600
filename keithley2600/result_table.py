# -*- coding: utf-8 -*-
#
# Copyright © keithley2600 Project Contributors
# Licensed under the terms of the MIT License
# (see keithley2600/__init__.py for details)

"""
Submodule defining classes to store, plot, and save measurement results.

"""

import os
import re
import warnings
from typing import List, Tuple, Union, Optional, Dict, Any, Callable, Sequence, Iterable

import numpy as np


def find_numbers(string: str) -> List[float]:

    fmt = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
    string_list = re.findall(fmt, string)
    float_list = [float(s) for s in string_list]

    return float_list


class ColumnTitle:
    """
    Class to hold a column title.

    :param name: Column name.
    :param unit: Column unit.
    :param unit_fmt: Formatting directive for units when generating string
        representations. By default, units are enclosed in square brackets (e.g.,
        "Gate voltage [V]").
    """

    def __init__(
        self, name: str, unit: Optional[str] = None, unit_fmt: str = "[{}]"
    ) -> None:

        self.name = name
        self.unit = "" if unit is None else unit
        self.unit_fmt = unit_fmt

    def has_unit(self) -> bool:
        return self.unit != ""

    def set_unit(self, unit: str) -> None:
        self.unit = unit

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(title='{self.name}', unit='{self.unit}')>"

    def __str__(self) -> str:
        if self.has_unit():
            return f"{self.name} {self.unit_fmt.format(self.unit)}"
        else:
            return self.name


class ResultTable:
    """
    Class that holds measurement data. All data is stored internally as a numpy array
    with the first index designating rows and the second index designating columns.

    Columns must have titles and can have units. It is possible to access the data in a
    column by its title in a dictionary type notation.

    :param list column_titles: List of column titles.
    :param list units: List of column units.
    :param data: Numpy array holding the data with the first index designating rows and
        the second index designating columns. If ``data`` is ``None``, an empty array
        with the required number of columns is created.
    :param params: Dictionary of measurement parameters.

    :Examples:

        Create a :class:`ResultTable` to hold current-vs-time data:

        >>> import time
        >>> import numpy as np
        >>> from  keithley2600 import ResultTable
        >>> # create dictionary of relevant measurement parameters
        >>> pars = {'recorded': time.asctime(), 'sweep_type': 'iv'}
        >>> # create ResultTable with two columns
        >>> rt = ResultTable(['Voltage', 'Current'], ['V', 'A'], params=pars)
        >>> # create a live plot of the data
        >>> fig = rt.plot(live=True)

        Create a :class:`Keithley2600` instance and record some data:

        >>> from  keithley2600 import Keithley2600
        >>> k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
        >>> for v in range(11):  # measure IV characteristics from 0 to 10 V
        ...     k.apply_voltage(k.smua, 10)
        ...     i = k.smua.measure.i()
        ...     rt.append_row([v, i])
        ...     time.sleep(1)

        Print a preview of data to the console:

        >>> print(rt)
        Voltage [V]   Current [A]
        0.0000e+00   1.0232e-04
        1.0000e+00   2.2147e-04
        2.0000e+00   3.6077e-04
        3.0000e+00   5.2074e-04
        4.0000e+00   6.9927e-04

        Save the recorded data to a tab-delimited text file:

        >>> rt.save('~/Desktop/stress_test.txt')

    """

    COMMENT = "# "
    DELIMITER = "\t"
    PARAM_DELIMITER = ": "
    LINE_BREAK = "\n"
    UNIT_FORMAT = "[{}]"

    def __init__(
        self,
        column_titles: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
        data: Optional[Sequence[float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:

        if column_titles is None:
            column_titles = []

        ncols = len(column_titles)

        if units is None:
            units = [""] * ncols

        self.titles = [
            ColumnTitle(n, u, self.UNIT_FORMAT) for n, u in zip(column_titles, units)
        ]

        if data is None:
            self.data = np.array([[]] * ncols).transpose()
        else:
            self.data = np.array(data)

        if params is None:
            self.params = {}
        else:
            self.params = params

    @property
    def nrows(self) -> int:
        """Number of rows of the ResultTable."""
        return self.data.shape[0]

    @property
    def ncols(self) -> int:
        """Number of columns of the ResultTable."""
        return len(self.titles)

    @property
    def shape(self) -> Tuple[int, int]:
        """A tuple representing the dimensionality of the ResultTable."""
        return self.data.shape[0], len(self.titles)

    @property
    def column_names(self) -> List[str]:
        """List of strings with column names."""
        return [title.name for title in self.titles]

    @column_names.setter
    def column_names(self, names_list: Sequence[str]) -> None:
        """Setter: List of strings with column names."""
        if not all(isinstance(x, str) for x in names_list):
            raise ValueError("All column names must be of type 'str'.")
        elif not len(names_list) == self.ncols:
            raise ValueError("Number of column names does not match number of columns.")

        for title, name in zip(self.titles, names_list):
            title.name = name

    @property
    def column_units(self) -> List[str]:
        """List of strings with column units."""
        return [title.unit for title in self.titles]

    @column_units.setter
    def column_units(self, units_list: Sequence[str]) -> None:
        """Setter: List of strings with column units."""
        if not all(isinstance(x, str) for x in units_list):
            raise ValueError("All column_units must be of type 'str'.")
        elif not len(units_list) == self.ncols:
            raise ValueError("Number of column_units does not match number of columns.")

        for title, unit in zip(self.titles, units_list):
            title.unit = unit

    def has_unit(self, col: Union[int, str]) -> bool:
        """
        Returns ``True`` column units have been set and ``False`` otherwise.

        :param col: Column index or name.
        :returns: ``True`` if column_units have been set, ``False`` otherwise.
        """
        if not isinstance(col, int):
            col = self.column_names.index(col)

        return self.titles[col].unit != ""

    def get_unit(self, col: Union[int, str]) -> str:
        """
        Get unit of column ``col``.

        :param col: Column index or name.
        :returns: Unit of column.
        """
        if not isinstance(col, int):
            col = self.column_names.index(col)

        return self.titles[col].unit

    def set_unit(self, col: Union[int, str], unit: str) -> None:
        """
        Set unit of column ``col``.

        :param col: Column index or name.
        :param unit: Unit string.
        """
        if not isinstance(col, int):
            col = self.column_names.index(col)

        self.titles[col].unit = unit

    def clear_data(self) -> None:
        """
        Clears all data.
        """
        self.data = np.array([[]] * self.ncols).transpose()

    def append_row(self, data: Sequence[float]) -> None:
        """
        Appends a single row to the data array.

        :param data: Sequence with the same number of elements as columns in the data
            array.
        """
        if not len(data) == self.ncols:
            raise ValueError(f"Length must match number of columns: {self.ncols}")

        self.data = np.append(self.data, [data], 0)

    def append_rows(self, data: Sequence[float]) -> None:
        """
        Appends multiple rows to the data array.

        :param data: List of lists or numpy array with dimensions matching the data
            array.
        """

        self.data = np.append(self.data, data, 0)

    def append_column(
        self, data: Sequence[float], name: str, unit: Optional[str] = None
    ) -> None:
        """
        Appends a single column to the data array.

        :param data: Sequence with the same number of elements as rows in the data
            array.
        :param name: Name of new column.
        :param unit: Unit of values in new column.
        """

        if self.data.size == 0:
            self.data = np.transpose([data])
        else:
            self.data = np.append(self.data, np.transpose([data]), 1)

        self.titles.append(ColumnTitle(name, unit, self.UNIT_FORMAT))

    def append_columns(
        self,
        data: Sequence[float],
        column_titles: List[str],
        units: Optional[List[str]] = None,
    ) -> None:
        """
        Appends multiple columns to data array.

        :param data: List of columns to append.
        :param column_titles: List of column titles (strings).
        :param units: List of units for new columns (strings).
        """
        if self.data.size == 0:
            self.data = np.transpose(data)
        else:
            self.data = np.append(self.data, np.transpose(data), 1)

        for name, unit in zip(column_titles, units):
            self.titles.append(ColumnTitle(name, unit, self.UNIT_FORMAT))

    def get_row(self, i: int) -> np.ndarray:
        """
        :param i: Index of row.
        :returns: Numpy array with data from row ``i``.
        """
        return self.data[i, :]

    def get_column(self, i: int) -> np.ndarray:
        """
        :param i: Index of column.
        :returns: Numpy array with data from column ``i``.
        """
        return self.data[:, i]

    def _column_title_string(self) -> str:
        """
        Creates column title string.

        :returns: String with column titles.
        """
        column_titles = [str(title) for title in self.titles]
        return self.DELIMITER.join(column_titles)

    def _parse_column_title_string(self, title_string: str) -> List[ColumnTitle]:
        """
        Parses a column title string.

        :param title_string: String to parse.
        :returns: List of column titles.
        """

        title_string = title_string.lstrip(self.COMMENT)

        # use only alphabetic characters in `unique`
        # otherwise `re.escape` may inadvertently escape them in Python < 3.7
        unique = "UNIQUESTRING"
        assert unique not in self.UNIT_FORMAT

        regexp_name = "(?P<name>.*) "
        regexp_unit = re.escape(self.UNIT_FORMAT.format(unique)).replace(
            unique, "(?P<unit>.*)"
        )

        strings = title_string.split(self.DELIMITER)

        titles = []

        for s in strings:
            m = re.search(regexp_name + regexp_unit, s)
            if m is None:
                titles.append(ColumnTitle(s, unit_fmt=self.UNIT_FORMAT))
            else:
                titles.append(
                    ColumnTitle(m.group("name"), m.group("unit"), self.UNIT_FORMAT)
                )

        return titles

    def _param_string(self) -> str:
        """
        Creates string containing all parameters from :attr:`params` as key, value pairs
        in separate lines marked as comments.

        :returns: Parameter string.
        """
        lines = []

        for key, value in self.params.items():
            lines.append(str(key) + self.PARAM_DELIMITER + str(value))

        return self.LINE_BREAK.join(lines)

    def _parse_param_string(self, header: str) -> Dict[str, Any]:
        """
        Parses comment section of _header to extract measurement parameters

        :returns: Dictionary containing measurement parameters.
        """

        params = {}

        lines = header.split(self.LINE_BREAK)

        for line in lines:
            if (
                line.startswith(self.COMMENT)
                and self.PARAM_DELIMITER in line
                and self.DELIMITER not in line
            ):
                contents = line.lstrip(self.COMMENT)
                key, value = contents.split(self.PARAM_DELIMITER)
                try:
                    params[key] = float(value)
                except ValueError:
                    if value in ["True", "true"]:
                        params[key] = True
                    elif value in ["False", "false"]:
                        params[key] = False
                    else:
                        params[key] = value

        return params

    def _header(self) -> str:
        """
        Outputs a full _header string with measurement parameters and column titles
        (including units).

        :returns: Header as string.
        """

        params_string = self._param_string()
        titles_string = self._column_title_string()

        return self.LINE_BREAK.join([params_string, titles_string])

    def _parse_header(self, header: str) -> Tuple[List[ColumnTitle], Dict[str, Any]]:
        """
        Parses a _header string . Returns list of :class:`ColumnTitle` objects and
        measurement parameters in dictionary.

        :param str header: Header to parse.
        :returns: Tuple with titles and params.
        """
        header = header.strip(self.LINE_BREAK)
        last_line = header.split(self.LINE_BREAK)[-1]

        titles = self._parse_column_title_string(last_line)
        params = self._parse_param_string(header)

        return titles, params

    def save(self, filename: str, ext: str = ".txt") -> None:
        """
        Saves the result table to a text file. The file format is:

        - The _header contains all measurement parameters as comments.
        - Column titles contain column_names and column_units of measured quantity.
        - Delimited columns contain the data.

        Files are saved with the specified extension (default: '.txt'). The classes
        default delimiters are used to separate columns and rows.

        :param filename: Path of file to save. Relative paths are interpreted with
            respect to the current working directory.
        :param ext: File extension. Defaults to '.txt'.
        """

        base_name = os.path.splitext(filename)[0]
        filename = base_name + ext

        np.savetxt(
            filename,
            self.data,
            delimiter=self.DELIMITER,
            newline=self.LINE_BREAK,
            header=self._header(),
            comments=self.COMMENT,
        )

    def save_csv(self, filename: str) -> None:
        """
        Saves the result table to a csv file. The file format is:

        - The _header contains all measurement parameters as comments.
        - Column titles contain column_names and column_units of measured quantity.
        - Comma delimited columns contain the data.

        Files are saved with the extension '.csv' and other extensions are
        overwritten.

        :param filename: Path of file to save. Relative paths are interpreted with
            respect to the current working directory.
        """

        old_delim = self.DELIMITER
        old_line_break = self.LINE_BREAK
        self.DELIMITER = ","
        self.LINE_BREAK = "\n"

        self.save(filename, ext=".csv")

        self.DELIMITER = old_delim
        self.LINE_BREAK = old_line_break

    def load(self, filename: str) -> None:
        """
        Loads data from csv or tab delimited tex file. The _header is searched for
        measurement parameters.

        :param filename: Absolute or relative path of file to load.
        """
        old_delim = self.DELIMITER
        old_line_break = self.LINE_BREAK

        base_name, ext = os.path.splitext(filename)

        if ext == ".csv":
            self.DELIMITER = ","
            self.LINE_BREAK = "\n"

        # read info string and header
        with open(filename) as f:
            lines = f.readlines()

        header_length = sum([l.startswith(self.COMMENT) for l in lines])
        header = "".join(lines[:header_length])

        self.titles, self.params = self._parse_header(header)

        # read data as 2D numpy array
        self.data = np.loadtxt(filename)

        self.DELIMITER = old_delim
        self.LINE_BREAK = old_line_break

    def plot(
        self,
        x_clmn: int = 0,
        y_clmns: Optional[List[str]] = None,
        func: Callable = lambda x: x,
        live: bool = False,
        **kwargs,
    ) -> "ResultTablePlot":
        """
        Plots the data. This method should not be called from a thread. The column
        containing the x-axis data is specified (defaults to first column), all other
        data is plotted on the y-axis. This method requires Matplotlib to be installed
        and accepts, in addition to the arguments documented here, the same keyword
        arguments as :func:`matplotlib.pyplot.plot`.

        Column titles are taken as legend labels. :func:`plot` tries to determine a
        common y-axis unit and name from all given labels.

        :param x_clmn: Integer or name of column containing the x-axis data.
        :type x_clmn: int or str
        :param y_clmns: List of column numbers or column names for y-axis data. If not
            given, all columns will be plotted against the x-axis column.
        :param func: Function to apply to y-data before plotting.
        :param live: If ``True``, update the plot when new data is added. Plotting will
            be carried out in the main (GUI) thread, therefore take care not to block
            the thread. This can be achieved for instance by adding data in a background
            thread which carries out the measurement, or by calling
            `matplotlib.pyplot.pause` after adding data to give the GUI time to update.
        :returns: :class:`ResultTablePlot` instance with Matplotlib figure.
        :raises ImportError: If import of matplotlib fails.
        """

        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting.")

        if live and not matplotlib.get_backend() == "Qt5Agg":
            warnings.warn(
                "'Qt5Agg' backend to Matplotlib is required for live plotting."
            )
            live = False

        plot = ResultTablePlot(self, x_clmn, y_clmns, func, live=live, **kwargs)

        return plot

    def __repr__(self) -> str:
        titles = [str(t) for t in self.titles]
        return f"<{self.__class__.__name__}(columns={titles}, data=array(...))>"

    def __str__(self) -> str:
        # print first 7 rows of ResultTable to console

        n = min(7, self.nrows)

        spacer = 3 * " "

        title_strings = [str(t) for t in self.titles]
        max_lengths = [max(len(ts), 11) for ts in title_strings]

        row_strings = [
            spacer.join([t.rjust(m) for t, m in zip(title_strings, max_lengths)])
        ]

        for row in self.data[0:n, :]:
            strings = ["{:.4e}".format(x) for x in row]
            strings = [s.rjust(m) for s, m in zip(strings, max_lengths)]
            row_strings.append(spacer.join(strings))

        return "\n".join(row_strings)

    # =============================================================================
    # Dictionary compatibility functions. This allows access to all columns of the
    # table with their names as keys.
    # =============================================================================

    def keys(self) -> List[str]:
        return self.column_names

    def has_key(self, key: str) -> bool:
        return self.__contains__(key)

    def values(self) -> List[np.ndarray]:
        return [self.get_column(i) for i in range(self.ncols)]

    def items(self) -> Iterable[Tuple[str, np.ndarray]]:
        return zip(self.keys(), self.values())

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Gets values in column with name ``key``.

        :param key: Column name.
        :returns: Column content as numpy array.
        """

        if key not in self.column_names:
            raise KeyError("No such column '{0}'.".format(key))
        if not isinstance(key, str):
            raise TypeError("Key must be of type 'str'.")

        return self.get_column(self.column_names.index(key))

    def __setitem__(self, key: str, value: Sequence[float]) -> None:
        """
        Sets values in column with name ``key``.

        :param str key: Column name.
        :param value: Sequence containing column values.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be of type 'str'.")

        if key not in self.column_names:
            self.append_column(value, name=key)
        else:
            self.data[:, self.column_names.index(key)] = np.transpose(value)

    def __delitem__(self, key: str) -> None:
        """
        Deletes column with name ``key``.

        :param str key:
        """
        i = self.column_names.index(key)
        self.data = np.delete(self.data, i, axis=1)
        self.titles.pop(i)

    def __iter__(self) -> "ResultTable":
        self._n = 0
        return self

    def __next__(self) -> str:
        if self._n + 1 <= self.ncols:
            r = self.column_names[self._n]
            self._n += 1
            return r
        else:
            raise StopIteration

    def __contains__(self, key) -> bool:
        return key in self.column_names

    def __len__(self) -> int:
        return self.ncols


class FETResultTable(ResultTable):
    """
    Class to handle, store and load transfer and output characteristic data of FETs.
    :class:`TransistorSweepData` inherits from :class:`ResultTable` and overrides the
    plot method.
    """

    @property
    def sweep_type(self) -> str:
        if "sweep_type" in self.params.keys():
            return self.params["sweep_type"]
        else:
            return ""

    @sweep_type.setter
    def sweep_type(self, sweep_type: str) -> None:
        self.params["sweep_type"] = sweep_type

    def plot(self, *args, **kwargs) -> "ResultTablePlot":
        """
        Plots the transfer or output curves. Overrides :func:`ResultTable.plot`.
        Absolute values are plotted, on a linear scale for output characteristics
        and a logarithmic scale for transfer characteristics. Takes the same arguments
        as :func:`ResultTable.plot`.

        :returns: :class:`ResultTablePlot` instance with Matplotlib figure.
        :raises ImportError: If import of matplotlib fails.
        """

        plot = ResultTable.plot(self, func=np.abs, *args, **kwargs)
        plot.ax.set_ylabel("I [A]")

        if self.sweep_type == "transfer":
            plot.ax.set_yscale("log")
        else:
            plot.ax.set_yscale("linear")

        return plot


class ResultTablePlot(object):
    """
    Plots the data from a given :class:`ResultTable` instance. Axes labels are
    automatically generated from column titles and units. This class requires Matplotlib
    to be installed. In addition to the arguments documented here, class:`ResultTable`
    accepts the same keyword arguments as :func:`matplotlib.pyplot.plot`.

    :param result_table: :class:`ResultTable` instance with data to plot.
    :type result_table: :class:`ResultTable`
    :param x_clmn: Integer or name of column containing the x-axis data.
    :type x_clmn: int or str
    :param y_clmns: List of column numbers or column names for y-axis data. If not
        given, all columns will be plotted against the x-axis column.
    :type y_clmns: list(int or str)
    :param func: Function to apply to y-data before plotting.
    :param live: If ``True``, update the plot when new data is added (default:
        ``False``). Plotting will be carried out in the main (GUI) thread, therefore
        take care not to block the thread. This can be achieved for instance by adding
        data in a background thread which carries out the measurement, or by calling
        `matplotlib.pyplot.pause` after adding data to give the GUI time to update.
    """

    def __init__(
        self,
        result_table: ResultTable,
        x_clmn: int = 0,
        y_clmns: List[Union[int, str]] = None,
        func: Callable = lambda x: x,
        live: bool = False,
        **kwargs,
    ):

        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting.")

        if live and not matplotlib.get_backend() == "Qt5Agg":
            warnings.warn(
                "'Qt5Agg' backend to Matplotlib is required for live plotting."
            )
            live = False

        # input processing
        self.result_table = result_table
        if self.result_table.ncols < 2:
            raise ValueError("'ResultTable' must at least contain two columns of data.")
        self.x_clmn = self._to_column_number(x_clmn)
        if y_clmns is None:
            self.y_clmns = list(range(0, self.result_table.ncols))
            self.y_clmns.remove(x_clmn)
        else:
            self.y_clmns = [self._to_column_number(c) for c in y_clmns]
        self.func = func

        # create plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        line_labels = []
        line_units = []
        self.lines = []

        x = self.result_table.data[:, self.x_clmn]

        for c in self.y_clmns:
            y = self.result_table.data[:, c]
            y = self.func(y)
            line = self.ax.plot(x, y, label=self.result_table.column_names[c], **kwargs)
            self.lines.append(line[0])

            line_labels.append(self.result_table.column_names[c])
            line_units.append(self.result_table.column_units[c])

        self.ax.set_xlabel(str(self.result_table.titles[x_clmn]))

        y_label = os.path.commonprefix(line_labels)
        y_unit = os.path.commonprefix(line_units)
        if y_unit == "":
            label_text = "%s" % y_label
        else:
            label_text = y_label + " " + self.result_table.UNIT_FORMAT.format(y_unit)
        self.ax.set_ylabel(label_text)

        self.ax.autoscale(enable=True, axis="x", tight=True)
        self.fig.tight_layout()

        self.fig.show()

        if live and matplotlib.get_backend() == "Qt5Agg":
            self._timer = self.fig.canvas.new_timer()
            self._timer.add_callback(self.update)
            self._timer.start(100)

    def show(self) -> None:
        """
        Shows the plot.
        """

        self.fig.show()

    def update(self) -> None:
        """
        Updates the plot with the data of the corresponding :class:`ResultTable`.
        This will be called periodically when :param:``live`` is ``True``.
        """

        x = self.result_table.data[:, self.x_clmn]

        for line, column in zip(self.lines, self.y_clmns):
            y = self.result_table.data[:, column]
            y = self.func(y)
            line.set_xdata(x)
            line.set_ydata(y)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _to_column_number(self, c: Union[str, int]) -> int:

        if not isinstance(c, int):
            c = self.result_table.column_names.index(c)

        return c
