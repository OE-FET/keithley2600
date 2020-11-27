#### v2.0.1

Fixes an error where a voltage sweep would end with an AttributeError: cannot set
property `busy`.

#### v2.0.0

This release completely overhauls how Keithley commands are generated. Instead of hard-
coding available commands for a particular series or model of Keithley, all available
commands are retrieved on demand from the Keithley itself. This is possible because the
Keithley's TSP scripts use the Lua programming language which allows such introspection.

The main disadvantage of this approach is that most Keithley attributes will only be
generated *after* connecting to an instrument. The main advantage is that all Keithley
commands of all models which use TSP are supported and support for any future commands
will be automatic. This removes the need to update the driver as the command set evolves,
barring changes to the syntax, and enables automatic support for models with different 
command sets or a different number of SMUs. Furthermore, there have been issues in the 
past with missing commands and constants due to oversight. Those will no longer occur.

The second major change is a switch from camel-case to snake-case for the public API. For
example, `Keithley2600.applyVoltage` has been renamed to `Keithley2600.apply_voltage`.

Other changes include:

- Type hints are used throughout.
- The Python syntax has been modernised for Python 3.6.
- The Keithley no longer beeps at the end of custom sweeps.
- Added API `Keithley2600.send_trigger` to send a trigger signal to the Keithley. This can
  be used to manually start a pre-programmed sweep.
- Added API `KeithleyClass.create_lua_attr` to create a new attribute in the Keithley
  namespace. Use `Keithley2600.create_lua_attr` to create global variables.
- Added API `KeithleyClass.delete_lua_attr` to delete an attribute from the Keithley
  namespace. Use `Keithley2600.delete_lua_attr` to delete global variables.
- `Keithley2600.connect()` now returns `True` if the connection was established and
  `False` otherwise.

#### v1.4.1

_Changed_:

- Replaced deprecated `visa` import with `pyvisa`.

#### v1.4.0

_Added_:

- Save time stamps with measurement data.

_Changed_:

- Renamed `ResultTablePlot.update_plot` to `ResultTablePlot.update`.
- Improved documentation of (live) plotting.
- Added `SENSE_LOCAL`, `SENSE_REMOTE` and `SENSE_CALA` to dictionary.

_Fixed_:

- Fixed explicitly defined methods such as `Keithley2600.applyVoltage` not appearing in
  dictionary.

#### v1.3.4

_Fixed_:

- Fixed a typo in the column labels of the dataset returned by `outputMeasurement`.

#### v1.3.3

_Added_:

- Added `__dir__` proprty to Keithley2600 and its classes to support autocompletion.
  The dictionary of commands is craeted from the Keithley reference manual.

_Changed_:

- Remember PyVisa connection settings which are passed as keyword arguments to
  `Keithley2600`. Previously, calling `Keithley2600.connect(...)` would revert to
  default settings.

_Fixed_:

- Explicitly set source mode in `Keithley2600.applyCurrent` and `Keithley2600.applyVoltage`.

#### v1.3.2

This release drops support for Python 2.7. Only Python 3.6 and higher are supported

_Fixed_:

- Fixed a bug in `rampToVoltage` where the target voltage would not be set correctly if
  it was smaller than the step size.

#### v1.3.1

_Added:_

- Optional argument `raise_keithley_errors`: If `True`, the Keithley's error queue will
  be checked after each command and any Keithley errors will be raised as Python errors.
  This causes significant communication overhead but facilitates the debugging of faulty
  scripts since an invalid command will raise a descriptive error instead of failing
  silently.

_Fixed:_

- Thread safety of communication with Keithley. `Keithley2600Base` now uses its own lock
  instead of relying on PyVisa's thread safety.

#### v1.3.0

This version includes some API changes and updates to the documentation and doc strings.

_Added:_

- Accept `range` (Python 2 and 3) and `xrange` (Python 2) as input for voltage sweep lists.

_Changed:_

- Methods `header` and `parse_header` of `ResultTable` are now private.
- Cleaned up and updated documentation.

_Removed:_

- Removed deprecated function `Keithley2600.clearBuffer()`. Use `buffer.clear()` and
  `buffer.clearcache()` instead where `buffer` is a Keithley buffer instance, such as
  `Keithley2600.smua.nvbuffer1`.

#### v1.2.2

_Added:_

- Added `shape` property to `ResultTable`.
- Added string representation of `ResultTable` which returns the first 7 rows as neatly
  formatted columns (similar to pandas dataframes).

#### v1.2.1

_Fixed:_

- Fixed a critical error when initializing and appending columns to an empty `ResultTable`
  instance.

#### v1.2.0

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

#### v1.1.1

_Fixed:_

- Fixed a thread safety bug: Fixed a bug that could cause the wrong result to be returned
  by a query when using `Keithley2600` from multiple threads.

#### v1.1.0

_Added:_

- Sphinx documentation.

#### v1.0.0

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
	  `TrasistorSweepData.load()` expects the parameter `sweep_type` to be present in
	  the _header and have one of the values: 'transfer' or 'output'.
	- Options to read and write in CSV format instead of tab-delimited columns are given.

	As a result, data files created by versions < 1.0.0 need to be modified as follows
	to be recognized:

	- Prepend '#' to the line with column titles.
	- Add the line '# sweep_type: type' to the _header where type can be 'transfer',
	  'output', or 'iv'.

_Removed:_

- `clearBuffers` method from `Keithley2600` has been deprecated. Clear the buffers
  directly with `buffer.clear()` instead, where `buffer` is a keithley buffer instance
  such as `k.smua.nvbuffer1`.

#### v0.3.0

_Added:_

- `Keithley2600` methods now accept `Keithley2600` objects as arguments, for instance,
  one can now write

  ```python
  # assume we have a Keithley2600 instance 'k'
  k.smua.measureiv(k.smua.nvbuffer1, k.smua.nvbuffer2)
  ```

  instead of needing to use their string representation:

  ```python
  k.smua.measureiv('smua.nvbuffer1', 'smua.nvbuffer2')
  ```
- Keyword arguments can now be given to `Keithley2600()` and will be passed on to the
  visa resource (e.g., `baud_rate=9600`).

_Changed:_

- Code simplifications resulting from the above.
- `k.readBuffer(buffer)` no longer clears the given buffer.
- When attempting to create a new instance of `Keithley2600` with the name VISA address
  as an existing instance, the existing instance is returned instead.

_Removed:_

- `k.clearBuffers(...)` now logs a deprecation warning and will be removed in v1.0.
  Clear the buffers directly with `buffer.clear()` instead.
