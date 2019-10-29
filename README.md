[![PyPI Release](https://img.shields.io/pypi/v/keithley2600.svg)](https://pypi.org/pypi/keithley2600/)
[![Downloads](https://pepy.tech/badge/keithley2600)](https://pepy.tech/project/keithley2600)
[![Build Status](https://travis-ci.com/OE-FET/keithley2600.svg?branch=master)](https://travis-ci.com/OE-FET/keithley2600)
[![Documentation Status](https://readthedocs.org/projects/keithley2600/badge/?version=latest)](https://keithley2600.readthedocs.io/en/latest/?badge=latest)

**Warning:** Version 1.3.2 only supports Python 3.6 and higher. Install version 1.3.1 if
you require support for Python 2.7.

# keithley2600
A full Python driver for the Keithley 2600 series of source measurement units.

## About
keithley2600 provides access to base functions and higher level functions such as IV
measurements, transfer and output curves, etc. Base commands replicate the functionality
and syntax from the Keithley's internal TSP functions, which have a syntax similar to
Python.

*Warning:*

There are currently only heuristic checks for allowed commands and arguments by the driver
itself. See the [Keithley 2600 reference manual](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8)
for all available commands. To enable command checking, set the keyword argument argument
`raise_keithley_errors = True` in the constructor. When `raise_keithley_errors` is set,
all invalid commands will be raised as Python errors. This is done by reading the
Keithley's error queue after every command and will therefore result in significant
communication overhead.

Almost all Keithley TSP commands can be used with this driver. Not supported are:

* All Keithley IV sweep commands. We implement our own in the `Keithley2600` class.
* Keithley TSP functions that have the same name as a Keithley TSP attribute (and vice
  versa). The driver cannot decide whether to handle them as a function call or as an
  attribute access. Currently, there is only one such case:
  - `io.output()` has been dropped because it conflicts with `smuX.source.output`, which
    is more commonly used.
* Keithley TSP commands that have the same name as built-in attributes of the driver.
  Currently, this is only:
  - `lan.trigger[N].connected`: conflicts with the attribute `Keithley2600.connected`.


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
data = k.readBuffer(k.smua.nvbuffer1)  # reads all entries from nvbuffer1 of SMUA
errs = k.readErrorQueue()  # gets all entries from error queue

k.setIntegrationTime(k.smua, 0.001)  # sets integration time in sec
k.applyVoltage(k.smua, 10)  # turns on and applies 10V to SMUA
k.applyCurrent(k.smub, 0.1)  # sources 0.1A from SMUB
k.rampToVoltage(k.smua, 10, delay=0.1, stepSize=1)  # ramps SMUA to 10V in steps of 1V

# sweep commands
k.voltageSweepSingleSMU(k.smua, range(0, 61), t_int=0.1,
                        delay=-1, pulsed=False)
k.voltageSweepDualSMU(smu1=k.smua, smu2=k.smub, smu1_sweeplist=range(0, 61),
                      smu2_sweeplist=range(0, 61), t_int=0.1, delay=-1, pulsed=False)
k.transferMeasurement( ... )
k.outputMeasurement( ... )
```

*Singleton behaviour:*

Once a Keithley2600 instance with a visa address `'address'` has been created, repeated
calls to `Keithley2600('address')` will return the existing instance instead of creating a
new one. This prevents the user from opening multiple connections to the same instrument
simultaneously and allows easy access to a Keithley2600 instance from different parts of a
program. For example:

```python
>>> from keithley2600 import Keithley2600
>>> k1 = Keithley2600('TCPIP0::192.168.2.121::INSTR')
>>> k2 = Keithley2600('TCPIP0::192.168.2.121::INSTR')
>>> print(k1 is k2)
True
```

*Data structures:*

The methods `voltageSweepSingleSMU` and `voltageSweepDualSMU` return lists with the
measured voltages and currents. The higher level commands `transferMeasurement` and
`outputMeasurement` return `ResultTable` objects which are somewhat similar to pandas
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
    k.applyVoltage(k.smua, 10)
    i = k.smua.measure.i()
    rt.append_row([v, i])

# save the data
rt.save('~/iv_curve.txt')
```

See the [documentation](https://keithley2600.readthedocs.io/en/latest/api/result_table.html)
for all available methods.

## Installation
Install the stable version from PyPi by running
```console
$ pip install keithley2600
```
or the latest development version from GitHub:
```console
$ pip install git+https://github.com/OE-FET/keithley2600
```

## System requirements

- Python 2.7 or 3.x

##  Documentation

See the Keithley 2600 reference manual
[here](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8)
for all available commands and arguments.
