# Keithley2600-driver
Full Python driver for the Keithley 2600 series source measurement units.

## About
Keithley driver with acccess to base functions and higher level functions such as IV measurements, tranfer and output curves, etc. Base commands replicate the functionality and syntax from the Keithley TSP functions, which have a syntax similar to python.

*Warning:*

There are currntly no checks for allowed arguments in the base commands. See the Keithley 2600 reference manual for all available commands and arguments. Almost all remotely accessible commands can be used with this driver. Not supported are:

* tspnet.excecute() # conflicts with Python's excecute command
* All Keithley IV sweep commands. We implement our own in the Keithley2600 class.

*Usage:*

Connect to the Keithley 2600 and perform some base commands:
```python
>>> from keithley_driver import Keithley2600
>>> k = Keithley2600('192.168.2.121')
>>> k.smua.source.output = k.smua.OUTPUT_ON  # turn on smuA
>>> k.smua.source.levelv = -40  # applies -40V to smuA
>>> volts = k.smua.measure.v()  # measures the smuA voltage
```

Some higher level commands defined in the driver:

```python
>>> k.clearBuffers() # clears measurement buffers of all SMUs
>>> data = k.readBuffer('smua.nvbuffer1') # reads out measurement data from buffer
>>> Vsweep, Isweep, Vfix, Ifix = k.voltageSweep(smu_sweep=k.smua, smu_fix=k.smub, VStart=0, VStop=-60,
	VStep=1, VFix=0, tInt=0.1, delay=-1, pulsed=True) # records IV curve
>>> k.rampToVoltage(smu=k.smua, targetVolt=60, delay=0.1, stepSize=1) # ramps to voltage
>>> data1 = k.outputMeasurement(...) # records output characteristics of a transistor
>>> data2 = k.transferMeasurement(...) # records transfer characteristics of a transistor
```


*Documentation:*

See the Keithley 2600 reference manual for all available commands and arguments.
