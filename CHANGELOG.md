#### v1.2.3-dev1 (2019-07-05):

_Added:_

- Accept `range` as input for voltage sweep lists in Python 3 (`xrange` in Python 2).

Changed:

- Methods `header` and `parse_header` of `ResultTable` are now private.

_Fixed:_

- Cleaned up and updated documentation.

#### v1.2.2 (2019-06-27):

_Added:_

- Added `shape` property to `ResultTable`.
- Added string representation of `ResultTable` which returns the first 7 rows as neatly
  formatted columns (similar to pandas dataframes).

#### v1.2.1 (2019-05-20):

_Fixed:_

- Fixed a critical error when initializing and appending columns to an emtpy `ResultTable`
  instance.

#### v1.2.0 (2019-05-16):

_Added:_

- New method `readErrorQueue` which returns a list of all errors in the Keithley's error
  queue.
- Support for Keithley TSP functions with multiple return values. Previously, only the
  first value would be returned.
- Added `ResultTablePlot` class to plot the data in a `ResultTable`.
- Added live plotting to `ResultTable` and its subclasses. Pass the keyword argument
  `live=True` to the `plot` method for the plot to update dynamically when new
  data is added.

_Changed:_

- Optimized I/O: Keithley function calls to only use a single `query` call instead of
  consecutive `query` and `read` calls.
- Emtpy strings returned by the Keithley will always be converted to `None`. This is
  necessary to enable the above change.
- Renamed `TransistorSweepData` to `FETResultTable`. Renamed `sweep_data` module to
  `result_table`.

_Removed:_

- Removed `IVSweepData`. The was no clear added value over using `ResultTable` directly.

#### v1.1.1 (2019-05-01):

_Fixed:_

- Fixed a thread safety bug: Fixed a bug that could cause the wrong result to be returned
  by a query when using `Keithley2600` from multiple threads.

#### v1.1.0 (2019-02-07):

_Added:_

- Sphinx documentation.

#### v1.0.0 (2019-01-17):

_Added:_

- Added the base class `ResultTable` to store, save and load tables of measurement data
  together with column titles, units, and measurement parameters. The data is stored
  internally as a 2D numpy array and can be accessed in a dictionary-type fashion with
  column names as keys. Additionally, `ResultTable` provides a basic plotting method
  using matplotlib.

_Changed:_

- `TrasistorSweepData` and `IVSweepData` now inherit from `ResultTable` and have been
   significantly simplified. Formats for saving and loading the data to files have
   slightly changed:

	- The line with column headers is now  marked as a comment and starts with '#'.
	- All given measurement parameters are saved in the file's _header. Specifically,
	  `TrasistorSweepData.load()` expects the parameter `sweep_type` to be present in the
	  _header and have one of the values: 'transfer' or 'output'.
	- Options to read and write in CSV format instead of tab-delimited columns are given.

	As a result, data files created by versions < 1.0.0 need to be modified as follows to
	be recognized:

	- Prepend '#' to the line with column titles.
	- Add the line '# sweep_type: type' to the _header where type can be 'transfer',
	  'output', or 'iv'.

_Removed:_

- `clearBuffers` method from `Keithley2600` has been deprecated. Clear the buffers
  directly with `buffer.clear()` instead, where `buffer` is a keithley buffer instance
  such as `k.smua.nvbuffer1`.

#### v0.3.0 (2018-11-13):

_Added:_

- `Keithley2600` methods now accept `Keithley2600` objects as arguments, for instance, one
  can now write

  ```python
  # assume we have a Keithley2600 instance 'k'
  k.smua.measureiv(k.smua.nvbuffer1, k.smua.nvbuffer2)
  ```

  instead of needing to use their string representation:

  ```python
  k.smua.measureiv('smua.nvbuffer1', 'smua.nvbuffer2')
  ```
- Keyword arguments can now be given to `Keithley2600()` and will be passed on to the visa
  resource (e.g., `baud_rate=9600`).

_Changed:_

- Code simplifications resulting from the above.
- `k.readBuffer(buffer)` no longer clears the given buffer.
- When attempting to create a new instance of `Keithley2600` with the name VISA address as
  an existing instance, the existing instance is returned instead.

_Removed:_

- `k.clearBuffers(...)` now logs a deprecation warning and will be removed in v1.0. Clear
  the buffers directly with `buffer.clear()` instead.
