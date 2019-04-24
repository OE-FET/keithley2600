# -*- coding: utf-8 -*-
#
# Copyright © keithley2600 Project Contributors
# Licensed under the terms of the MIT License
# (see keithley2600/__init__.py for details)

"""
Submodule defining classes to store, plot, and save measurement results.

"""

import sys
import os
import re
import time
import numpy as np

PY2 = sys.version[0] == '2'

if not PY2:
    basestring = str  # in Python 3


def find_numbers(string):

    fmt = r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?'
    string_list = re.findall(fmt, string)
    float_list = [float(s) for s in string_list]

    return float_list


class ColumnTitle(object):
    """
    Object to hold a column title.

    :ivar str name: String containing column name.
    :ivar str unit: String containing column unit.
    :ivar str unit_fmt: Formatting directive for units when generating
        string representation.
    """

    def __init__(self, name, unit=None, unit_fmt='[{}]'):

        self.name = name
        self.unit = '' if unit is None else unit
        self.unit_fmt = unit_fmt

    def has_unit(self):
        return self.unit != ''

    def set_unit(self, unit):
        self.unit = unit

    def __repr__(self):
        return "<{0}(title='{1}', unit='{2}')>".format(
            self.__class__.__name__, self.name, self.unit)

    def __str__(self):
        if self.has_unit():
            return self.name + ' ' + self.unit_fmt.format(self.unit)
        else:
            return self.name


# noinspection PyTypeChecker
class ResultTable(object):
    """
    Class that holds measurement data. All data is stored internally in a numpy
    array.

    Columns must have names, to designate the measurement variable, and
    can have units. It is possible to access columns by their
    names in a dictionary type notation.

    :ivar list names: List of column names (strings).
    :ivar list units: List of column units (strings).
    :ivar data: Numpy array holding the data.
    :ivar dict params: Dictionary of measurement parameters.

    :Example:

        Create a :class:`ResultTable` to hold current-vs-time data, for instance for
        a bias stress test of a device:

        >>> import time
        >>> import numpy as np
        >>> from  keithley2600 import ResultTable
        >>> # create dictionary of relevant measurement parameters
        >>> par_dict = {'Voltage': 10, 'Recorded': time.asctime()}
        >>> # create ResultTable with two columns
        >>> table = ResultTable(['Time', 'Current'], ['sec', 'A'], par_dict)

        Create a :class:`Keithley2600` instance and record some data:

        >>> from  keithley2600 import Keithley2600
        >>> k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
        >>> k.applyVoltage(k.smua, 10)  # apply a voltage
        >>> for t in range(120):  # measure each second, for 2 min
        ...     i = k.smua.measure.i()
        ...     table.append_row([t, i])  # add values to ResultTable
        ...     time.sleep(1)

        Save and plot the recorded data:

        >>> table.save('~/Desktop/stress_test.txt')  # save the ResultTable
        >>> table.plot()  # plot the data
        >>> table['Current']  # look at the data
        array([0.01, ..., 0.01])

    """

    COMMENT = '# '
    DELIMITER = '\t'
    PARAM_DELIMITER = ': '
    LINE_BREAK = '\n'
    UNIT_FORMAT = '[{}]'

    def __init__(self, names=None, units=None, data=None, params=None):

        if names is None:
            names = []

        if units is None:
            units = [''] * len(names)

        self.titles = [ColumnTitle(n, u, self.UNIT_FORMAT) for n, u in zip(names, units)]

        if data is None:
            self.data = None
        else:
            self.data = np.array(data)

        if params is None:
            self.params = {}
        else:
            self.params = params

    @property
    def nrows(self):
        if self.data is None:
            return 0
        else:
            return self.data.shape[0]

    @property
    def ncols(self):
        return len(self.titles)

    @property
    def column_names(self):
        return [title.name for title in self.titles]

    @column_names.setter
    def column_names(self, names_list):
        if not all(isinstance(x, basestring) for x in names_list):
            raise ValueError("All column names must be of type 'str'.")
        elif not len(names_list) == self.ncols:
            raise ValueError('Number of column names does not match number of columns.')

        for title, name in zip(self.titles, names_list):
            title.name = name

    @property
    def column_units(self):
        return [title.unit for title in self.titles]

    @column_units.setter
    def column_units(self, units_list):
        if not all(isinstance(x, basestring) for x in units_list):
            raise ValueError("All column_units must be of type 'str'.")
        elif not len(units_list) == self.ncols:
            raise ValueError('Number of column_units does not match number of columns.')

        for title, unit in zip(self.titles, units_list):
            title.unit = unit

    def has_unit(self, col):
        """
        Returns `True` if column_units of column ``col`` have been set, `False` otherwise.

        :param col: Column index or name.

        :returns: `True` if column_units have been set, `False` otherwise.
        :rtype: bool
        """
        if not isinstance(col, int):
            col = self.column_names.index(col)

        return self.titles[col].unit != ''

    def get_unit(self, col):
        """
        Get unit of column ``col``.

        :param col: Column index or name (int or str)
        :returns: Unit string.
        :rtype: str
        """
        if not isinstance(col, int):
            col = self.column_names.index(col)

        return self.titles[col].unit

    def set_unit(self, col, unit):
        """
        Set unit of column ``col``.

        :param col: Column index or name.
        :param str unit: Unit string.
        """
        if not isinstance(col, int):
            col = self.column_names.index(col)

        self.titles[col].unit = unit

    def clear_data(self):
        """
        Clears all data.
        """
        self.data = None

    def append_row(self, data):
        """
        Appends a row to data array.

        :param data: Iterable with the same number of elements as columns in the data
            array.
        """
        if not len(data) == self.ncols:
            raise ValueError('Length must match number of columns: %s' % self.ncols)

        if self.data is None:
            self.data = np.array([data])
        else:
            self.data = np.append(self.data, [data], 0)

    def append_rows(self, data):
        """
        Appends multiple rows to data array.

        :param data: List of lists or numpy array with dimensions matching the data array.
        """

        if self.data is None:
            self.data = np.array(data)
        else:
            self.data = np.append(self.data, data, 0)

    def append_column(self, data, name, unit=None):
        """
        Appends a new column to data array.

        :param data: Iterable with the same number of elements as rows in the data array.
        :param str name: Column name.
        :param str unit: Unit of values in new column.
        """

        if self.data is None:
            self.data = np.array(np.transpose([data]))
        else:
            self.data = np.append(self.data, np.transpose([data]), 1)

        self.titles.append(ColumnTitle(name, unit, self.UNIT_FORMAT))

    def append_columns(self, data, names, units=None):
        """
        Appends multiple columns to data array.

        :param list data: List of columns to append.
        :param list names: Column column_names.
        :param list units: Units for new columns.
        """

        if self.data is None:
            self.data = np.array(np.transpose(data))
        else:
            self.data = np.append(self.data, np.transpose(data), 1)

        for name, unit in zip(names, units):
            self.titles.append(ColumnTitle(name, unit, self.UNIT_FORMAT))

    def get_row(self, i):
        """
        Returns row i as a 1D numpy array.
        """
        return self.data[i, :]

    def get_column(self, i):
        """
        Returns column i as a 1D numpy array.
        """
        return self.data[:, i]

    def _column_title_string(self):
        """
        Creates column title string.

        :returns: String with column titles.
        :rtype: str
        """
        column_titles = [str(title) for title in self.titles]
        return self.DELIMITER.join(column_titles)

    def _parse_column_title_string(self, title_string):
        """
        Parses a column title string.

        :param str title_string: String to parse.

        :returns: List of :class:`ColumnTitle` instances.
        """

        title_string = title_string.lstrip(self.COMMENT)

        # use only alphabetic characters in `unique`
        # otherwise `re.escape` may inadvertently escape them in Python < 3.7
        unique = 'UNIQUESTRING'
        assert unique not in self.UNIT_FORMAT

        regexp_name = '(?P<name>.*) '
        regexp_unit = re.escape(self.UNIT_FORMAT.format(unique)).replace(unique,
                                                                         '(?P<unit>.*)')

        strings = title_string.split(self.DELIMITER)

        titles = []

        for s in strings:
            m = re.search(regexp_name + regexp_unit, s)
            if m is None:
                titles.append(ColumnTitle(s, unit_fmt=self.UNIT_FORMAT))
            else:
                titles.append(ColumnTitle(m.group('name'), m.group('unit'),
                                          self.UNIT_FORMAT))

        return titles

    def _param_string(self):
        """
        Creates string containing all parameters from :attr:`params` as key, value pairs
        in separate lines marked as comments.

        :returns: Parameter string.
        :rtype: str
        """
        lines = []

        for key, value in self.params.items():
            lines.append(str(key) + self.PARAM_DELIMITER + str(value))

        return self.LINE_BREAK.join(lines)

    def _parse_param_string(self, header):
        """
        Parses comment section of header to extract measurement parameters

        :returns: Dictionary containing measurement parameters.
        :rtype: dict
        """

        params = {}

        lines = header.split(self.LINE_BREAK)

        for line in lines:
            if (line.startswith(self.COMMENT) and self.PARAM_DELIMITER in line
                    and self.DELIMITER not in line):
                contents = line.lstrip(self.COMMENT)
                key, value = contents.split(self.PARAM_DELIMITER)
                try:
                    params[key] = float(value)
                except ValueError:
                    if value in ['True', 'true']:
                        params[key] = True
                    elif value in ['False', 'false']:
                        params[key] = False
                    else:
                        params[key] = value

        return params

    def header(self):
        """
        Outputs full header with comment section containing measurement parameters and
        column titles including units.

        :returns: Header as string.
        :rtype: str
        """

        params_string = self._param_string()
        titles_string = self._column_title_string()

        return self.LINE_BREAK.join([params_string, titles_string])

    def parse_header(self, header):
        """
        Parses header. Returns list of :class:`ColumnTitle` objects and measurement
        parameters in dictionary.

        :param str header: Header to parse.
        :returns: Tuple with titles and params.
        :rtype: (str, str)
        """
        header = header.strip(self.LINE_BREAK)
        last_line = header.split(self.LINE_BREAK)[-1]

        titles = self._parse_column_title_string(last_line)
        params = self._parse_param_string(header)

        return titles, params

    def save(self, filename, ext='.txt'):
        """
        Saves the result table to a text file. The file format is:

        - The header contains all measurement parameters as comments.
        - Column titles contain column_names and column_units of measured quantity.
        - Delimited columns contain the data.

        Files are saved with the specified extension (default: .txt). The
        classes default delimiters are used to separate columns and rows.

        :param str filename: Path of file to save. Relative paths are
            interpreted with respect to the current working directory.
        :param str ext: File extension (default: .txt)
        """

        base_name = os.path.splitext(filename)[0]
        filename = base_name + ext

        np.savetxt(filename, self.data, delimiter=self.DELIMITER, newline=self.LINE_BREAK,
                   header=self.header(), comments=self.COMMENT)

    def save_csv(self, filename):
        """
        Saves the result table to a csv file. The file format is:

        - The header contains all measurement parameters as comments.
        - Column titles contain column_names and column_units of measured quantity.
        - Comma delimited columns contain the data.

        Files are saved with the extension '.csv' and other extensions are
        overwritten.

        :param str filename: Path of file to save. Relative paths are
            interpreted with respect to the current working directory.
        """

        old_delim = self.DELIMITER
        old_line_break = self.LINE_BREAK
        self.DELIMITER = ','
        self.LINE_BREAK = '\n'

        self.save(filename, ext='.csv')

        self.DELIMITER = old_delim
        self.LINE_BREAK = old_line_break

    def load(self, filename):
        """
        Loads data from csv or tab delimited tex file. The header is searched
        for measurement parameters.

        :param str filename: Absolute or relative path of file to load.
        """
        old_delim = self.DELIMITER
        old_line_break = self.LINE_BREAK

        base_name, ext = os.path.splitext(filename)

        if ext == '.csv':
            self.DELIMITER = ','
            self.LINE_BREAK = '\n'

        # read info string and header
        with open(filename) as f:
            lines = f.readlines()

        header_length = sum([l.startswith(self.COMMENT) for l in lines])
        header = ''.join(lines[:header_length])

        self.titles, self.params = self.parse_header(header)

        # read data as 2D numpy array
        self.data = np.loadtxt(filename)

        self.DELIMITER = old_delim
        self.LINE_BREAK = old_line_break

    def plot(self, x_clmn=0, y_clmn=None, func=lambda x: x, **kwargs):
        """
        Plots the data. This method should not be called from a thread.
        The column containing the x-axis data is specified (defaults to first
        column), all other data is plotted on the y-axis. Keyword arguments are
        passed on to pyplot.

        Column titles are taken as legend labels. `plot` tries to determine a
        common y-axis unit and name from all given labels.

        :param x_clmn: Integer or name of column containing the x-axis data.
        :param y_clmn: List of column numbers or column_names for y-axis data. If not
            given, all columns will be plotted (excluding the x-axis column).
        :param func: Function to apply to y-data before plotting.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('Warning: Install matplotlib to support plotting.')
            return

        if self.ncols == 0:
            return

        if not isinstance(x_clmn, int):
            x_clmn = self.column_names.index(x_clmn)

        xdata = self.data[:, x_clmn]

        fig = plt.figure()

        ax = fig.add_subplot(111)

        if y_clmn is None:
            ydata = np.delete(self.data, [x_clmn], axis=1)
            ydata = func(ydata)
            lines = ax.plot(xdata, ydata, **kwargs)
            line_labels = self.column_names[0:x_clmn] + self.column_names[x_clmn + 1:]
            line_units = self.column_units[0:x_clmn] + self.column_units[x_clmn + 1:]
            ax.legend(lines, line_labels)
        else:
            line_labels = []
            line_units = []
            for c in y_clmn:
                if not isinstance(c, int):
                    c = self.column_names.index(c)
                ydata = self.data[:, c]
                ydata = func(ydata)
                ax.plot(xdata, ydata, label=self.column_names[c], **kwargs)

                line_labels.append(self.column_names[c])
                line_units.append(self.column_units[c])

        ax.set_xlabel(str(self.titles[x_clmn]))

        y_label = os.path.commonprefix(line_labels)
        y_unit = os.path.commonprefix(line_units)
        if y_unit == '':
            ax.set_ylabel('%s' % y_label)
        else:
            ax.set_ylabel(y_label + ' ' + self.UNIT_FORMAT.format(y_unit))

        ax.autoscale(enable=True, axis='x', tight=True)
        fig.tight_layout()

        self.setup_plot(fig, ax)

        fig.show()

        return fig, ax

    def setup_plot(self, fig, ax):
        """
        This method does nothing by default, but can be overwritten by the child
        class in order to set up custom options for plotting.

        :param fig: Matplotlib figure instance.
        :param ax: Matplotlib axes instance.
        """
        pass

    def __repr__(self):
        titles = [str(t) for t in self.titles]
        return '<{0}(columns={1}, data=array(...))>'.format(
                self.__class__.__name__, str(titles))

# =============================================================================
# Dictionary compatibility functions. This allows access to all columns of the
# table with their names as keys.
# =============================================================================

    def keys(self):
        return self.column_names

    def has_key(self, key):
        return self.__contains__(key)

    def values(self):
        return [self.get_column(i) for i in range(self.ncols)]

    def items(self):
        return zip(self.keys(), self.values())

    def __getitem__(self, key):
        """
        Gets values in column with name `key`.
        :param str key: Column name.
        :returns: Column content as numpy array.
        """

        if key not in self.column_names:
            raise KeyError("No such column '{0}'.".format(key))
        if not isinstance(key, str):
            raise TypeError("Key must be of type 'str'.")

        return self.get_column(self.column_names.index(key))

    def __setitem__(self, key, value):
        """
        Sets values in column with name `key`.

        :param str key: Column name.
        :param value: Iterable containing column values.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be of type 'str'.")

        if key not in self.column_names:
            self.append_column(value, name=key)
        else:
            self.data[:, self.column_names.index(key)] = np.transpose(value)

    def __delitem__(self, key):
        """
        Deletes column with name `key`.

        :param str key:
        """
        i = self.column_names.index(key)
        self.data = np.delete(self.data, i, axis=1)
        self.titles.pop(i)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n + 1 <= self.ncols:
            r = self.column_names[self._n]
            self._n += 1
            return r
        else:
            raise StopIteration

    def __contains__(self, key):
        return key in self.column_names

    def __len__(self):
        return self.ncols


class IVSweepData(ResultTable):
    """
    Class to store, load, save, and plot data from a simple IV sweep.
    :class:`IVSweepData` inherits form :class:`ResultTable` but provides
    the fixed columns 'Voltage' and 'Current' with units 'V' and 'A', respectively.

    The following attributes are new or overwritten:

    :cvar str sweep_type: Describes the sweep type, defaults to 'iv'.

    """

    sweep_type = 'iv'

    def __init__(self, v=None, i=None, params=None):

        if (i is None) or (v is None):
            data = None
        else:
            data = np.transpose([v, i])

        if params is None:
            params = {}

        names = ['Voltage', 'Current']
        units = ['V', 'A']
        params['sweep_type'] = self.sweep_type
        params['recorded'] = time.localtime()

        super(self.__class__, self).__init__(names, units, data, params)

    def append(self, v, i):
        """
        Appends list-like objects with voltage and current values to data.

        :param v: List (or list-like object) containing voltage values.
        :param i: List (or list-like object) containing current values.
        """

        assert len(i) == len(v)
        self.append_rows(np.transpose([v, i]))


class TransistorSweepData(ResultTable):
    """
    Class to handle, store and load transfer and output characteristic data of FETs.
    :class:`TransistorSweepData` inherits from :class:`ResultTable` and overrides the
    plot method.

    The following attributes are new or overwritten:
    """

    @property
    def sweep_type(self):
        return self.params['sweep_type']

    @sweep_type.setter
    def sweep_type(self, value):
        self.params['sweep_type'] = value

    def stepped_voltage_list(self):
        """
        Get voltage steps of transfer / output characteristics. This returns
        the drain voltages steps for transfer curve data and gate voltage steps
        for output curve data.

        :returns: Voltage steps in transfer / output characteristics.
        :rtype: set
        """

        return set(find_numbers(self._column_title_string()))

    def n_steps(self):
        """
        Gets the number of steps in transfer or output curve.

        :returns: Number of drain voltage steps for transfer curves or number of
            gate voltage steps for output curves.
        :rtype: int
        """
        return len(self.stepped_voltage_list())

    def plot(self, *args, **kwargs):
        """
        Plots the transfer or output curves. Overrides :func:`ResultTable.plot`.
        Absolute values are plotted, on a linear scale for output characteristics
        and a logarithmic scale for transfer characteristics. All arguments are passed
        on to :func:`ResultTable.plot`.
        """

        if self.sweep_type == 'transfer':
            fig, ax = super(self.__class__, self).plot(func=np.abs, *args, **kwargs)
            ax.set_yscale('log')
            ax.set_ylabel('I [A]')

        elif self.sweep_type == 'output':
            fig, ax = super(self.__class__, self).plot(func=np.abs, *args, **kwargs)
            ax.set_yscale('linear')
            ax.set_ylabel('I [A]')
