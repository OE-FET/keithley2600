# keithley2600
Full Python driver for the Keithley 2600 series source measurement units.

## About
Keithley driver with access to base functions and higher level functions such as IV measurements, transfer and output curves, etc. Base commands replicate the functionality and syntax from the Keithley TSP functions, which have a syntax similar to python.

*Warning:*

There are currently no checks for allowed arguments in the base commands. See the [Keithley 2600 reference manual](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8) for all available commands and arguments. Almost all remotely accessible commands can be used with this driver. Not supported are:

* tspnet.excecute() # conflicts with Python's execute command
* lan.trigger[N].connected # conflicts with the connected attribute of Keithley2600Base
* All Keithley IV sweep commands. We implement our own in the Keithley2600 class.

*Usage:*

Connect to the Keithley 2600 and perform some base commands:
```python
>>> from keithley2600 import Keithley2600
>>> k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
>>> k.smua.source.output = k.smua.OUTPUT_ON  # turn on smuA
>>> k.smua.source.levelv = -40  # sets smuA source level to -40V without turning the smu on or off
>>> volts = k.smua.measure.v()  # measures and returns the smuA voltage
>>> k.smua.measure.v(k.smua.nvbuffer1)  # measures the smuA voltage, stores the result in the given buffer
>>> k.smua.nvbuffer1.clear()  # clears nvbuffer1 of smuA
```
Higher level commands defined in the driver:

```python
>>> data = k.readBuffer(k.smua.nvbuffer1)  # reads entries from nvbuffer1 of smuA
>>> k.setIntegrationTime(k.smua, 0.001) # in sec

>>> k.applyVoltage(k.smua, -60) # applies -60V to k.smua
>>> k.applyCurrent(k.smub, 0.1) # sources 0.1A from k.smub
>>> k.rampToVoltage(k.smua, 10, delay=0.1, stepSize=1) # ramps k.smua to 10V in 1V steps
>>> i = k.smua.measure.i() # mesures current at smuA

>>> k.voltageSweepSingleSMU(smu=k.smua, smu_sweeplist=list(range(0, 61)),
                            tInt=0.1, delay=-1, pulsed=False)  # records single SMU IV curve
>>> k.voltageSweepDualSMU(smu1=k.smua, smu2=k.smub, smu1_sweeplist=list(range(0, 61)),
                          smu2_sweeplist=list(range(0, 61)), tInt=0.1, delay=-1,
                          pulsed=False)  # records dual SMU IV curve
```


## Installation
Download or clone the repository. Install the package by running
```console
$ pip install git+https://github.com/OE-FET/keithley2600
```

##  Documentation

See the Keithley 2600 reference manual [here](https://www.tek.com/keithley-source-measure-units/smu-2600b-series-sourcemeter-manual-8) for all available commands and arguments.
