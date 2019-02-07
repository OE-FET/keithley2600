#### v1.0.0 (2019-01-17):
_Added:_

- Added the base class `ResultTable` to store, save and load tables of measurement data togther with column titles, units, and measurement parameters. The data is stored internally as a 2D numpy array and can be accessed in a dictionary-type fashion with column names as keys. Additionaly, `ResultTable` provides a basic plotting method using matplotlib.

_Changed:_

- `TrasistorSweepData` and `IVSweepData` now inherit from `ResultTable` and have been significantly simplified. Formats for saving and loading the data to files have slightly changed:

	- The line with column headers is now  marked as a comment and starts with '#'.
	- All given measurement parameters are saved in the file's header. Specifically, `TrasistorSweepData.load()` expects the parameter     `sweep_type` to be present in the header and
	  have one of the values: 'transfer' or 'output'.
	- Options to read and write in CSV format instead of tab-delimited columns are given.

	As a result, data files created by versions < 1.0.0 need to be modified as follows to be recoginzed:

	- Prepend '#' to the line with column titles.
	- Add the line '# sweep_type: type' to the header where type can be 'transfer', 'output', or 'iv'.

_Removed:_

- `clearBuffers` method from `Keithley2600` has been deprecated. Clear the buffers directly with `buffer.clear()` instead, where `buffer` is a keithley buffer instance such as `k.smua.nvbuffer1`.

#### v0.3.0 (2018-11-13):
_Added:_

- `Keithley2600` methods now accecpt `Keithley2600` objects as arguments, for instance, one can now write
  ```python
  # assume we have a Keithley2600 instance 'k'
  k.smua.measureiv(k.smua.nvbuffer1, k.smua.nvbuffer2)
  ```
  instead of needing to use their string representation:
  ```python
  k.smua.measureiv('smua.nvbuffer1', 'smua.nvbuffer2')
  ```
- Keyword aruments can now be given to `Keithley2600()` and will be passed on to the visa resource (e.g., `baud_rate=9600`)

_Changed:_

- Code simplifications resulting from the above.
- `k.readBuffer(buffer)` no longer clears the given buffer.
- When attempting to create a new instance of `Keithley2600` with the name VISA address as an existing instance, the existing instance is returned instead.

_Removed:_

- `k.clearBuffers(...)` now logs a deprecation warning and will be removed in v1.0. Clear the buffers directly with `buffer.clear()` instead.
 