[![PyPI Release](https://img.shields.io/pypi/v/keithley2600.svg)](https://pypi.org/pypi/keithley2600/)
[![Downloads](https://pepy.tech/badge/keithley2600)](https://pepy.tech/project/keithley2600)
[![Build Status](https://travis-ci.com/OE-FET/keithley2600.svg?branch=master)](https://travis-ci.com/OE-FET/keithley2600)
[![Documentation Status](https://readthedocs.org/projects/keithley2600/badge/?version=latest)](https://keithley2600.readthedocs.io/en/latest/?badge=latest)

# keithley2600

A full Python driver for the Keithley 2600B series of source measurement units. An
accompanying GUI is provided by the sister project
[keithleygui](https://github.com/OE-FET/keithleygui). Documentation is available at
[https://keithley2600.readthedocs.io](https://keithley2600.readthedocs.io).

## About

This driver provides access to base commands and higher level functions such as IV
measurements, transfer and output curves, etc. Base commands replicate the functionality
and syntax from the Keithley's internal TSP Lua functions. This is possible because the
Lua programming language has a very limited syntax which can be represented by a subset
of Python syntax.

All Keithley commands are dynamically queried from the Keithley itself after a
successful connection. This means that essentially all Keithley instruments which use
TSP scripting are supported and any commands introduced in the future will be recognised
automatically (barring changes to the Lua syntax itself). Please refer to the respective
reference manuals for a list of commands available on a particular model, for instance the
[Keithley 2600B reference manual](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8).

This dynamic approach however means that most attributes will only be defined after
connecting to an instrument. Several higher level functions for current-voltage sweeps
are defined by the driver itself and may use functionality which is not available on
some  models. They have been tested with a Keithley 2612B.

**Warning**: There are currently no checks for allowed arguments by the driver itself.
Passing invalid arguments to a Keithley command will fail silently in Python but will
display an error message on the Keithley itself. To enable command checking, set the
keyword argument `raise_keithley_errors = True` in the constructor. When set, most
Keithley errors will be raised as Python errors. This is done by reading the Keithley's
error queue after every command and will therefore result in some communication
overhead.

## Installation

Install the stable version from PyPi by running
```console
$ pip install keithley2600
```
or the latest development version from GitHub:
```console
$ pip install git+https://github.com/OE-FET/keithley2600
```

## Usage

Connect to the Keithley 2600 and perform some base commands:
```python
from keithley2600 import Keithley2600

k = Keithley2600('TCPIP0::192.168.2.121::INSTR')

k.smua.source.output = k.smua.OUTPUT_ON   # turn on SMUA
k.smua.source.levelv = -40  # sets SMUA source level to -40V
v = k.smua.measure.v()  # measures and returns the SMUA voltage
i = k.smua.measure.i()  # measures current at smuA

k.smua.measure.v(k.smua.nvbuffer1)  # measures the voltage, stores the result in buffer
k.smua.nvbuffer1.clear()  # clears nvbuffer1 of SMUA
```
Higher level commands defined in the driver:

```python
data = k.read_buffer(k.smua.nvbuffer1)  # reads all entries from nvbuffer1 of SMUA
errs = k.read_error_queue()  # gets all entries from error queue

k.set_integration_time(k.smua, 0.001)  # sets integration time in sec
k.apply_voltage(k.smua, 10)  # turns on and applies 10V to SMUA
k.apply_current(k.smub, 0.1)  # sources 0.1A from SMUB
k.ramp_to_voltage(k.smua, 10, delay=0.1, stepSize=1)  # ramps SMUA to 10V in steps of 1V

# sweep commands
k.voltage_sweep_single_smu(
	k.smua, range(0, 61), t_int=0.1, delay=-1, pulsed=False
)
k.voltage_sweep_dual_smu(
	smu1=k.smua, smu2=k.smub, smu1_sweeplist=range(0, 61), smu2_sweeplist=range(0, 61),
	t_int=0.1, delay=-1, pulsed=False
)
k.transfer_measurement( ... )
k.output_measurement( ... )
```

*Singleton behaviour:*

Once a Keithley2600 instance with a visa address `'address'` has been created, repeated
calls to `Keithley2600('address')` will return the existing instance instead of creating
a new one. This prevents the user from opening multiple connections to the same
instrument simultaneously and allows easy access to a Keithley2600 instance from
different parts of a program. For example:

```python
>>> from keithley2600 import Keithley2600
>>> k1 = Keithley2600('TCPIP0::192.168.2.121::INSTR')
>>> k2 = Keithley2600('TCPIP0::192.168.2.121::INSTR')
>>> print(k1 is k2)
True
```

*Data structures:*

The methods `voltage_sweep_single_smu` and `voltage_sweep_dual_smu` return lists with
the measured voltages and currents. The higher level commands `transfer_measurement` and
`output_measurement` return `ResultTable` objects which are somewhat similar to pandas
dataframes but include support for column units. `ResultTable` stores the measurement
data internally as a numpy array and provides information about column titles and units.
It also provides a dictionary-like interface to access columns by name, methods to load
and save the data to text files, and live plotting of the data (requires matplotlib).

For example:
```python
import time
from  keithley2600 import Keithley2600, ResultTable

k = Keithley2600('TCPIP0::192.168.2.121::INSTR')

# create ResultTable with two columns
rt = ResultTable(column_titles=['Voltage', 'Current'], units=['V', 'A'],
                 params={'recorded': time.asctime(), 'sweep_type': 'iv'})

# create live plot which updates as data is added
rt.plot(live=True)

# measure some currents
for v in range(0, 20):
    k.apply_voltage(k.smua, 10)
    i = k.smua.measure.i()
    rt.append_row([v, i])

# save the data
rt.save('~/iv_curve.txt')
```

See the [documentation](https://keithley2600.readthedocs.io/en/latest/api/result_table.html)
for all available methods.

## Backend selection

keithley2600 uses [PyVISA](https://pyvisa.readthedocs.io/) to connect to instruments.
PyVISA supports both proprietray IVI libraries such as NI-VISA, Keysight VISA, R&S VISA,
tekVISA etc. and the purely Python backend [PyVISA-py](https://pyvisa-py.readthedocs.io/en/latest/).
You can select a specific backend by giving its path to the `Keithley2600` constructor
in the `visa_library` argument. For example:

```python
from  keithley2600 import Keithley2600

k = Keithley2600('TCPIP0::192.168.2.121::INSTR', visa_library='/usr/lib/libvisa.so.7')
```

keithley2600 defaults to using the PyVISA-py backend, selected by `visa_library='@py'`,
since this is only a pip-install away. If you pass an empty string, keithley2600 will
use an installed IVI library if it can find one in standard locations or fall back to
PyVISA-py otherwise.

You can find more information about selecting the backend in the
[PyVISA docs](https://pyvisa.readthedocs.io/en/latest/introduction/configuring.html).

## System requirements

- Python 3.6 and higher

##  Documentation

* API documentation of keithley2600: [https://keithley2600.readthedocs.io/en/latest/](https://keithley2600.readthedocs.io/en/latest/)

* Keithley 2600 reference manual with all commands: [https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8)
