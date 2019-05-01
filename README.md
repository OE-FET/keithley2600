[![PyPi Release](https://img.shields.io/pypi/v/keithley2600.svg)](https://pypi.org/project/keithley2600/)
[![Build Status](https://travis-ci.com/OE-FET/keithley2600.svg?branch=master)](https://travis-ci.com/OE-FET/keithley2600)
[![Documentation Status](https://readthedocs.org/projects/keithley2600/badge/?version=latest)](https://keithley2600.readthedocs.io/en/latest/?badge=latest)

# keithley2600
A full Python driver for the Keithley 2600 series of source measurement units.

## About
Keithley driver with access to base functions and higher level functions such as IV
measurements, transfer and output curves, etc. Base commands replicate the functionality
and syntax from the Keithley's internal TSP functions, which have a syntax similar to Python.

*Warning:*

There are currently only heuristic checks for allowed arguments in the base commands. See the
[Keithley 2600 reference manual](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8)
for all available commands and arguments. Almost all remotely accessible commands can be
used with this driver. Not supported are:

* `lan.trigger[N].connected`: conflicts with the connected attribute of `Keithley2600Base`.
* `io.output()`: conflicts with `smuX.source.output` attribute
* All Keithley IV sweep commands. We implement our own in the `Keithley2600` class.

*Usage:*

Connect to the Keithley 2600 and perform some base commands:
```python
>>> from keithley2600 import Keithley2600
>>> k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
>>> k.smua.source.output = k.smua.OUTPUT_ON  # turn on smuA
>>> k.smua.source.levelv = -40 # sets smuA source level to -40V
>>> volts = k.smua.measure.v() # measures and returns the smuA voltage
>>> k.smua.measure.v(k.smua.nvbuffer1) # measures the voltage, stores the result in buffer
>>> k.smua.nvbuffer1.clear() # clears nvbuffer1 of smuA
```
Higher level commands defined in the driver:

```python
>>> data = k.readBuffer(k.smua.nvbuffer1) # reads entries from nvbuffer1 of smuA
>>> k.setIntegrationTime(k.smua, 0.001) # in sec

>>> k.applyVoltage(k.smua, -60) # applies -60V to k.smua
>>> k.applyCurrent(k.smub, 0.1) # sources 0.1A from k.smub
>>> k.rampToVoltage(k.smua, 10, delay=0.1, stepSize=1) # ramps k.smua to 10V in 1V steps
>>> i = k.smua.measure.i() # measures current at smuA

>>> k.voltageSweepSingleSMU(smu=k.smua, smu_sweeplist=list(range(0, 61)),
...                         t_int=0.1, delay=-1, pulsed=False) # records an IV curve
>>> k.voltageSweepDualSMU(smu1=k.smua, smu2=k.smub, smu1_sweeplist=list(range(0, 61)),
...                       smu2_sweeplist=list(range(0, 61)), t_int=0.1, delay=-1,
...                       pulsed=False) # records dual SMU IV curve
```


## Installation
Install the stable version from PyPi by running
```console
$ pip install keithley2600
```
or the latest development version from GitHub:
```console
$ pip install git+https://github.com/OE-FET/keithley2600
```

##  Documentation

See the Keithley 2600 reference manual
[here](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8)
for all available commands and arguments.
