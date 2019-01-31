# -*- coding: utf-8 -*-
#
# Copyright © keithley2600 Project Contributors
# Licensed under the terms of the MIT License
# (see keithley2600/__init__.py for details)

"""
Core driver with the low level functions.
"""

# system imports
from __future__ import absolute_import, division, print_function
import sys
import visa
import logging
import threading
import numpy as np
import time

# local import
from keithley2600.keithley_doc import (CONSTANTS, FUNCTIONS, PROPERTIES,
                                       CLASSES, PROPERTY_LISTS)
from keithley2600.sweep_data import TransistorSweepData

PY2 = sys.version[0] == '2'
logger = logging.getLogger(__name__)

if not PY2:
    basestring = str  # in Python 3


def log_to_screen(level=logging.DEBUG):
    log_to_stream(None, level)  # sys.stderr by default


def log_to_stream(stream_output, level=logging.DEBUG):
    logger.setLevel(level)
    ch = logging.StreamHandler(stream_output)
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)


class MagicPropertyList(object):
    """Mimics a Keithley TSP property list

    Class which mimics a Keithley TSP property list and can be dynamically
    created. It forwards all calls to the _read method of the parent class and
    assignments to the _write method. Arbitrary values can be assigned, as
    long as _write can handle them.

    This class is designed to look like a  Keithley TSP "attribute" list,
    forward function calls to the Keithley, and return the results.

    """

    def __init__(self, name, parent):
        if not isinstance(name, basestring):
            raise ValueError('First argument must be of type str.')
        self._name = name
        self._parent = parent

    def __getitem__(self, i):
        """Gets i-th item: query item from parent class

        Args:
            i: An integer item number

        Returns:
            Result from _query call of parent class.
        """
        new_name = '%s[%s]' % (self._name, i)
        return self._query(new_name)

    def __setitem__(self, i, value):
        """Sets i-th item: set item at parent class

        Args:
            i: An integer item number
            value: An input object that can be accepted by parent class.

        Returns:
            None.
        """
        value = self._convert_input(value)
        new_name = '%s[%s] = %s' % (self._name, i, value)
        self._write(new_name)

    def __iter__(self):
        return self

    def _write(self, value):
        """Forward _write calls to parent class."""
        self._parent._write(value)

    def _query(self, value):
        """Forward _query calls to parent class."""
        return self._parent._query(value)

    def _convert_input(self, value):
        """Forward _convert_input calls to parent class."""
        try:
            return self._parent._convert_input(value)
        except AttributeError:
            return value

    def getdoc(self):
        """Prevent pydoc from trying to document this class. This could
        conflict with on-demand creation of attributes."""
        pass


class MagicFunction(object):
    """Mimics a Keithley TSP function

    Class which mimics a function and can be dynamically created. It forwards
    all calls to the _query method of the parent class and returns the result
    from _query. Calls accept arbitrary arguments, as long as _query can
    handle them.

    This class is designed to look like a Keithley TSP function, forward
    function calls to the Keithley, and return the results.

    """

    def __init__(self, name, parent):
        if not isinstance(name, basestring):
            raise ValueError('First argument must be of type str.')
        self._name = name
        self._parent = parent

    def __call__(self, *args, **kwargs):
        """Pass on calls to `self.prent._write`, store result in variable.
        Querying results from function calls directly may result in
        a VisaIOError if the function does not return anything."""

        # convert incompatible arguments, return all arguments as tuple
        args = tuple(self._parent._convert_input(a) for a in args)
        # remove outside brackets and all quotation marks
        args_string = str(args).strip("(),").replace("'", "")
        # pass on calls to self._write as string representing function call
        self._parent._write('result = %s(%s)' % (self._name, args_string))
        # query for result in second call
        return self._parent._query('result')


class MagicClass(object):
    """Mimics a TSP command group

    Class which dynamically creates new attributes on access. These can be
    functions, properties, or other classes.

    MagicClass need the strings in FUNCTIONS and PROPERTIES to determine if the
    accessed attribute should behave like a function or property. Otherwise, it
    is assumed to be a new class.

    Attribute setters and getters are forwarded to _write and _query functions
    from the parent class. New functions are created as instances of
    MagicFunction, new classes are created as instances of MagicClass.

    MagicClass is designed to mimic a Keithley TSP command group with
    functions, attributes, and subordinate command groups.

    USAGE:
        inst = MagicClass('keithley')
        inst.reset() - Dynamically creates a new attribute 'reset' as an instance
                       of MagicFunction, then calls it.
        inst.beeper  - Dynamically creates new attribute 'beeper' and sets it to
                       a new MagicClass instance.
        inst.beeper.enable - Fakes the property 'enable' of 'beeper'
                             with _write as setter and _query as getter.

    """

    _name = ''
    _parent = None

    def __init__(self, name, parent=None):
        if not isinstance(name, basestring):
            raise ValueError('First argument must be of type str.')
        self._name = name
        self._parent = parent

    def __getattr__(self, attr_name):
        """Custom getter

        Get attributes as usual if they exist. Otherwise, fall back to
        `self.__get_global_handler`.
        """
        try:
            try:
                # check if attribute already exists. return attr if yes.
                return object.__getattr__(self, attr_name)
            except AttributeError:
                # check if key already exists. return value if yes.
                return self.__dict__[attr_name]
        except KeyError:
            # handle if not
            return self.__get_global_handler(attr_name)

    def __get_global_handler(self, attr_name):
        """Custom getter

        Create attribute as MagicClass, MagicFunction or MagicPropertyList
        instance if it is an expected Keithley TSP command group, function or
        property list. Query and return value if attribute corresponds to a
        Keithley TSP constant. Otherwise raise standard AttributeError.

        Args:
            attr_name: Attribute name.

        Returns:
            Instance or MagicClass, MagicFunction or MagicPropertyList.

        Raises:
            AttributeError if attribute is not expected.
        """

        # create callable sub-class for new attr
        new_name = '%s.%s' % (self._name, attr_name)
        new_name = new_name.strip('.')

        if attr_name in FUNCTIONS:
            handler = MagicFunction(new_name, parent=self)
            self.__dict__[new_name] = handler

        elif attr_name in PROPERTY_LISTS:
            handler = MagicPropertyList(new_name, parent=self)

        elif attr_name in PROPERTIES or attr_name in CONSTANTS:
            if new_name in PROPERTY_LISTS:
                handler = MagicPropertyList(new_name, parent=self)
            else:
                handler = self._query(new_name)

        elif attr_name in CLASSES:
            handler = MagicClass(new_name, parent=self)
            self.__dict__[new_name] = handler

        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self), attr_name)
                )

        return handler

    def __setattr__(self, attr_name, value):
        """Custom setter

        Forward setting commands to `self._write` for expected Keithley TSP
        attributes. Otherwise use default setter.

        Args:
            attr_name: Attribute name.
            value: Value to set.

        Returns:
            None.

        Raises:
            ValueError if trying to write a value to read-only Keithley
            attributes.
        """
        if attr_name in PROPERTIES:
            value = self._convert_input(value)
            self._write('%s.%s = %s' % (self._name, attr_name, value))
        elif attr_name in CONSTANTS:
            raise ValueError('%s.%s is read-only.' % (self._name, attr_name))
        else:
            object.__setattr__(self, attr_name, value)
            self.__dict__[attr_name] = value

    def _write(self, value):
        """Forward _write calls to parent class."""
        self._parent._write(value)

    def _query(self, value):
        """Forward _query calls to parent class."""
        return self._parent._query(value)

    def _convert_input(self, value):
        """Forward _convert_input calls to parent class."""
        try:
            return self._parent._convert_input(value)
        except AttributeError:
            return value

    def __getitem__(self, i):
        """Return new MagicClass instance for every item."""
        new_name = '%s[%s]' % (self._name, i)
        new_class = MagicClass(new_name, parent=self)
        return new_class

    def __iter__(self):
        return self

    def getdoc(self):
        """Prevent pydoc from trying to document this class. This could
        conflict with on-demand creation of attributes."""
        pass


class KeithleyIOError(Exception):
    pass


class Keithley2600Base(MagicClass):
    """Keithley2600 driver

    Keithley driver for base functions. It replicates the functionality and
    syntax from the Keithley TSP commands, which have a syntax similar to
    python. Attributes are created on-access if they correspond to Keithley TSP
    type commands.

    Warning:
        There are currently no checks for allowed arguments in the base
        commands. See the Keithley 2600 reference manual for all available
        commands and arguments. Almost all remotely accessible commands can be
        used with this driver. NOT SUPPORTED ARE:
             * `tspnet.execute()` - conflicts with Python's execute command
             * `lan.trigger[N].connected` - conflicts with the connected attribute of
                Keithley2600Base
             * All Keithley IV sweep commands. We implement our own in the
               Keithley2600 class.

    Example:
        >>> keithley = Keithley2600Base('TCPIP0::192.168.2.121::INSTR')
        >>> keithley.smua.measure.v()  # measures the smuA voltage
        >>> keithley.smua.source.levelv = -40  # applies -40V to smuA

    Documentation:
        See the Keithley 2600 reference manual for all available commands and
        arguments.

    Attributes:
        _lock (threading.RLock): Lock to prevent simultaneous calls to
            the Keithley.
        connection (visa resource): Attribute holding reference to the actual
            connection.
        connected (bool): Attribute to hold info if connected.
    """

    _lock = threading.RLock()
    connection = None
    connected = False
    busy = False

    # input types that will be accepted as TSP lists by keithley
    TO_TSP_LIST = (list, np.ndarray, tuple, set)
    # maximum length of lists send to keithley
    CHUNK_SIZE = 50

    def __init__(self, visa_address, visa_library='@py', **kwargs):
        """Initializes driver, connects to Keithley

        Args:
            visa_address: Visa address of Keithley containing connection type
                and system address, e.g., "TCPIP0::192.168.2.121::INSTR". See
                NI-VISA for PyVisa for documentation
            visa_library: Visa backend used by PyVisa. Can be an empty string
                for NI-VISA backend, a path to the visa library, or "@py" for
                the py-visa-py backend.
        """

        MagicClass.__init__(self, name='', parent=self)
        self._name = ''  # visa_address will

        self.abort_event = threading.Event()

        self.visa_address = visa_address
        self.visa_library = visa_library

        # open visa resource manager with selected library / backend
        self.rm = visa.ResourceManager(self.visa_library)
        # connect to keithley
        self.connect(**kwargs)

    def __repr__(self):
        return '<%s(%s)>' % (type(self).__name__, self.visa_address)

# =============================================================================
# Connect to keithley
# =============================================================================

    def connect(self, **kwargs):
        """
        Connects to Keithley and opens pyvisa API.
        """
        connection_error = OSError if PY2 else ConnectionError
        # noinspection PyBroadException
        try:
            self.connection = self.rm.open_resource(self.visa_address, **kwargs)
            self.connection.read_termination = '\n'
            self.connected = True
            logger.debug('Connected to Keithley at %s.' % self.visa_address)
        except ValueError:
            self.connection = None
            self.connected = False
            raise
        except connection_error:
            logger.info('Connection error. Please check that ' +
                        'no other program is connected.')
            self.connection = None
            self.connected = False
        except AttributeError:
            logger.info('Invalid VISA address %s.' % self.visa_address)
            self.connection = None
            self.connected = False
        except Exception:
            logger.info('Could not connect to Keithley at %s.' % self.visa_address)
            self.connection = None
            self.connected = False

    def disconnect(self):
        """ Disconnect from Keithley """
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                self.connected = False
                del self.connection
                logger.debug('Disconnected from Keithley at %s.' % self.visa_address)
            except AttributeError:
                self.connected = False
                pass

# =============================================================================
# Define I/O
# =============================================================================

    def _write(self, value):
        """
        Writes text to Keithley. Input must be a string.
        """
        logger.debug('write: %s' % value)

        if self.connection:
            self.connection.write(value)
        else:
            raise KeithleyIOError(
                'No connection to keithley present. Try to call connect().')

    def _query(self, value):
        """
        Queries and expects response from Keithley. Input must be a string.
        """
        logger.debug('write: print(%s)' % value)

        if self.connection:
            with self._lock:
                r = self.connection.query('print(%s)' % value)
                logger.debug('read: %s' % r)

            return self.parse_response(r)
        else:
            raise KeithleyIOError(
                'No connection to keithley present. Try to call connect().')

    @staticmethod
    def parse_response(string):
        try:
            r = float(string)
        except ValueError:
            if string == 'nil':
                r = None
            elif string == 'true':
                r = True
            elif string == 'false':
                r = False
            else:
                r = string

        return r

    def _convert_input(self, value):
        """ Convert bool to lower case string and list / tuples to comma
        delimited string enclosed by curly brackets."""
        if isinstance(value, bool):
            # convert bool True to string 'true'
            value = str(value).lower()
        elif isinstance(value, self.TO_TSP_LIST):
            # convert some iterables to a TSP type list '{1,2,3,4}'
            value = '{%s}' % ', '.join(map(str, value))
        elif isinstance(value, MagicClass):
            # convert keithley object to string with its name
            value = value._name

        return value


class Keithley2600(Keithley2600Base):
    """Keithley2600 driver with high level functionality

    Keithley driver with access to base functions and higher level functions
    such as IV measurements, transfer and output curves, etc. Base command
    replicate the functionality and syntax from the Keithley TSP functions,
    which have a syntax similar to python.

    Warning:
        There are currently no checks for allowed arguments in the base
        commands. See the Keithley 2600 reference manual for all available
        commands and arguments. Almost all remotely accessible commands can be
        used with this driver. NOT SUPPORTED ARE:
             * tspnet.execute() # conflicts with Python's execute command
             * All Keithley IV sweep commands. We implement our own here.

    Example:
        Base commands from keithley TSP:

        >>> k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
        >>> volts = k.smua.measure.v()  # measures and returns the smuA voltage
        >>> k.smua.source.levelv = -40  # sets source level of smuA
        >>> k.smua.nvbuffer1.clear()  # clears nvbuffer1 of smuA

        New mid-level commands:

        >>> data = k.readBuffer(k.smua.nvbuffer1)
        >>> k.setIntegrationTime(k.smua, 0.001) # in sec

        >>> k.applyVoltage(k.smua, -60) # applies -60V to smuA
        >>> k.applyCurrent(k.smub, 0.1) # sources 0.1A from smuB
        >>> k.rampToVoltage(k.smua, 10, delay=0.1, step_size=1)

        >>> # voltage sweeps, single and dual SMU
        >>> k.voltageSweepSingleSMU(smu=k.smua, smu_sweeplist=list(range(0, 61)),
        ...                         t_int=0.1, delay=-1, pulsed=False)
        >>> k.voltageSweepDualSMU(smu1=k.smua, smu2=k.smub,
        ...                       smu1_sweeplist=list(range(0, 61)),
        ...                       smu2_sweeplist=list(range(0, 61)),
        ...                       t_int=0.1, delay=-1, pulsed=False)

        New high-level commands:

        >>> data1 = k.outputMeasurement(...) # records output curve
        >>> data2 = k.transferMeasurement(...) # records transfer curve

    Attributes:
        SMU_LIST (list): List containing strings of all smu column_names.
    """

    SMU_LIST = ['smua', 'smub']

    def __init__(self, visa_address, visa_library='@py', **kwargs):
        Keithley2600Base.__init__(self, visa_address, visa_library, **kwargs)

    def __repr__(self):
        return '<%s(%s)>' % (type(self).__name__, self.visa_address)

    def _check_smu(self, smu):
        """Check if selected smu is indeed present."""
        assert smu._name.split('.')[-1] in self.SMU_LIST

    @staticmethod
    def _get_smu_string(smu):
        return smu._name.split('.')[-1]

# =============================================================================
# Define lower level control functions
# =============================================================================

    @staticmethod
    def readBuffer(buffer):
        """
        Reads buffer values and returns them as a list. This can be done more
        quickly by "readings = buffer.readings" but such a call may fail due to
        I/O limitations of the keithley if the readings list is too long.

        Args:
            buffer: A keithley buffer instance.

        Returns:
            A list with buffer readings.
        """
        list_out = []
        for i in range(0, int(buffer.n)):
            list_out.append(buffer.readings[i+1])

        return list_out

    def clearBuffer(self, smu):
        """
        Clears buffer of a given smu. This function has been deprecated.
        """

        raise DeprecationWarning(
            "'clearBuffer()' has been deprecated. Please use 'buffer.clear()' and " +
            "'buffer.clearcache()' instead, where 'buffer' is a Keithley2600 buffer " +
            "instance such as 'k.smua.nvbuffer1'.")

    def setIntegrationTime(self, smu, t_int):
        """
        Sets the integration time of SMU for measurements in sec.

        Args:
            smu: A keithley smu instance.
            t_int (float): Integration time in sec. Value must be between 0.001
                and 25 power line cycles.
        Raises:
            ValueError for too short or too long integration times.
        """

        self._check_smu(smu)

        # determine number of power-line-cycles used for integration
        freq = self.localnode.linefreq
        nplc = t_int * freq

        if nplc < 0.001 or nplc > 25:
            raise ValueError('Integration time must be between 0.001 and 25 ' +
                             'power line cycles of 1/(%s Hz).' % freq)
        smu.measure.nplc = nplc

    def applyVoltage(self, smu, voltage):
        """
        Turns on the specified SMU and applies a voltage.
        """

        self._check_smu(smu)

        smu.source.output = smu.OUTPUT_ON
        smu.source.levelv = voltage

    def applyCurrent(self, smu, curr):
        """
        Turns on the specified SMU and sources a current.
        """
        self._check_smu(smu)

        smu.source.leveli = curr
        smu.source.output = smu.OUTPUT_ON

    def rampToVoltage(self, smu, target_volt, delay=0.1, step_size=1):
        """
        Ramps up the voltage of the specified SMU. Beeps when done.

        Args:
            smu: SMU to ramp.
            target_volt (float): Target gate voltage.
            step_size (float): Size of the voltage ramp steps in Volts.
            delay (float): Delay between steps in sec.
        """

        self._check_smu(smu)

        smu.source.output = smu.OUTPUT_ON

        # get current voltage
        vcurr = smu.source.levelv
        if vcurr == target_volt:
            return

        self.display.smua.measure.func = self.display.MEASURE_DCVOLTS
        self.display.smub.measure.func = self.display.MEASURE_DCVOLTS

        step = np.sign(target_volt - vcurr) * abs(step_size)

        for v in np.arange(vcurr, target_volt + step, step):
            smu.source.levelv = v
            smu.measure.v()
            time.sleep(delay)

        target_volt = smu.measure.v()
        logger.info('Gate voltage set to Vg = %s V.' % round(target_volt))

        self.beeper.beep(0.3, 2400)

    def voltageSweepSingleSMU(self, smu, smu_sweeplist, t_int, delay, pulsed):
        """
        Sweeps voltage at one SMU. Measures and returns current and voltage during sweep.

        Args:
            smu (keithley smu object): 1st SMU to be swept.
            smu_sweeplist (list): Voltages to sweep through, must be a list,
                tuple, or numpy array.
            t_int (float): Integration time per data point.
            delay (float): Settling delay before measurement.
            pulsed (bool): True or False for pulsed or continuous sweep.

        Returns:
            v_smu (list): Voltages measurement during the sweep in Volts.
            i_smu (list): Currents measurement during the sweep in Amperes.
        """

        # input checks
        self._check_smu(smu)

        # set state to busy
        self.busy = True
        # Define lists containing results. If we abort early, we have something to return.
        v_smu, i_smu = [], []

        if self.abort_event.is_set():
            self.busy = False
            return v_smu, i_smu

        # setup smu to sweep through list on trigger
        # send sweep_list over in chunks if too long
        if len(smu_sweeplist) > self.CHUNK_SIZE:
            self._write('mylist = {}')
            for num in smu_sweeplist:
                self._write('table.insert(mylist, %s)' % num)
            smu.trigger.source.listv('mylist')
        else:
            smu.trigger.source.listv(smu_sweeplist)

        smu.trigger.source.action = smu.ENABLE

        # CONFIGURE INTEGRATION TIME FOR EACH MEASUREMENT
        self.setIntegrationTime(smu, t_int)

        # CONFIGURE SETTLING TIME FOR GATE VOLTAGE, I-LIMIT, ETC...
        smu.measure.delay = delay
        smu.measure.autorangei = smu.AUTORANGE_ON

        # smu.trigger.source.limiti = 0.1

        smu.source.func = smu.OUTPUT_DCVOLTS

        # 2-wire measurement (use SENSE_REMOTE for 4-wire)
        # smu.sense = smu.SENSE_LOCAL

        # clears SMU buffers
        smu.nvbuffer1.clear()
        smu.nvbuffer2.clear()

        smu.nvbuffer1.clearcache()
        smu.nvbuffer2.clearcache()

        # display current values during measurement
        self.display.smua.measure.func = self.display.MEASURE_DCAMPS
        self.display.smub.measure.func = self.display.MEASURE_DCAMPS

        # SETUP TRIGGER ARM AND COUNTS
        # trigger count = number of data points in measurement
        # arm count = number of times the measurement is repeated (set to 1)

        npts = len(smu_sweeplist)
        smu.trigger.count = npts

        # SET THE MEASUREMENT TRIGGER ON BOTH SMU'S
        # Set measurement to trigger once a change in the gate value on
        # sweep smu is complete, i.e., a measurement will occur
        # after the voltage is stepped.
        # Both channels should be set to trigger on the sweep smu event
        # so the measurements occur at the same time.

        # enable smu
        smu.trigger.measure.action = smu.ENABLE

        # measure current and voltage on trigger, store in buffer of smu
        smu.trigger.measure.iv(smu.nvbuffer1, smu.nvbuffer2)

        # initiate measure trigger when source is complete
        smu.trigger.measure.stimulus = smu.trigger.SOURCE_COMPLETE_EVENT_ID

        # SET THE ENDPULSE ACTION TO HOLD
        # Options are SOURCE_HOLD AND SOURCE_IDLE, hold maintains same voltage
        # throughout step in sweep (typical IV sweep behavior). idle will allow
        # pulsed IV sweeps.

        if pulsed:
            end_pulse_action = 0  # SOURCE_IDLE
        elif not pulsed:
            end_pulse_action = 1  # SOURCE_HOLD
        else:
            raise TypeError("'pulsed' must be of type 'bool'.")

        smu.trigger.endpulse.action = end_pulse_action

        # SET THE ENDSWEEP ACTION TO HOLD IF NOT PULSED
        # Output voltage will be held after sweep is done!

        smu.trigger.endsweep.action = end_pulse_action

        # SET THE EVENT TO TRIGGER THE SMU'S TO THE ARM LAYER
        # A typical measurement goes from idle -> arm -> trigger.
        # The 'trigger.event_id' option sets the transition arm -> trigger
        # to occur after sending *trg to the instrument.

        smu.trigger.arm.stimulus = self.trigger.EVENT_ID

        # Prepare an event blender (blender #1) that triggers when
        # the smua enters the trigger layer or reaches the end of a
        # single trigger layer cycle.

        # triggers when either of the stimuli are true ('or enable')
        self.trigger.blender[1].orenable = True
        self.trigger.blender[1].stimulus[1] = smu.trigger.ARMED_EVENT_ID
        self.trigger.blender[1].stimulus[2] = smu.trigger.PULSE_COMPLETE_EVENT_ID

        # SET THE smu SOURCE STIMULUS TO BE EVENT BLENDER #1
        # A source measure cycle within the trigger layer will occur when
        # either the trigger layer is entered (termed 'armed event') for the
        # first time or a single cycle of the trigger layer is complete (termed
        # 'pulse complete event').

        smu.trigger.source.stimulus = self.trigger.blender[1].EVENT_ID

        # PREPARE AN EVENT BLENDER (blender #2) THAT TRIGGERS WHEN BOTH SMU'S
        # HAVE COMPLETED A MEASUREMENT.
        # This is needed to prevent the next source measure cycle from occurring
        # before the measurement on both channels is complete.

        self.trigger.blender[2].orenable = True  # triggers when both stimuli are true
        self.trigger.blender[2].stimulus[1] = smu.trigger.MEASURE_COMPLETE_EVENT_ID

        # SET THE SMU ENDPULSE STIMULUS TO BE EVENT BLENDER #2
        smu.trigger.endpulse.stimulus = self.trigger.blender[2].EVENT_ID

        # TURN ON smu
        smu.source.output = smu.OUTPUT_ON

        # INITIATE MEASUREMENT
        # prepare SMUs to wait for trigger
        smu.trigger.initiate()

        # send trigger
        self._write('*trg')

        # CHECK STATUS BUFFER FOR MEASUREMENT TO FINISH
        # Possible return values:
        # 6 = smua and smub sweeping
        # 4 = only smub sweeping
        # 2 = only smua sweeping
        # 0 = neither smu sweeping

        status = 0
        while status == 0:  # while loop that runs until the sweep begins
            status = self.status.operation.sweeping.condition
            time.sleep(0.1)

        while status > 0:  # while loop that runs until the sweep ends
            status = self.status.operation.sweeping.condition
            time.sleep(0.1)

        # EXTRACT DATA FROM SMU BUFFERS
        i_smu = self.readBuffer(smu.nvbuffer1)
        v_smu = self.readBuffer(smu.nvbuffer2)

        smu.nvbuffer1.clear()
        smu.nvbuffer2.clear()

        smu.nvbuffer1.clearcache()
        smu.nvbuffer2.clearcache()

        self.busy = False

        return v_smu, i_smu

    def voltageSweepDualSMU(self, smu1, smu2, smu1_sweeplist, smu2_sweeplist, t_int,
                            delay, pulsed):
        """
        Sweeps voltages at two SMUs. Measures and returns current and voltage during
        sweep.

        Args:
            smu1 (keithley smu object): 1st SMU to be swept.
            smu2 (keithley smu object): 2nd SMU to be swept.
            smu1_sweeplist: List of voltages to sweep at smu1 (can be a numpy
                 array, list or tuple).
            smu2_sweeplist: List of voltages to sweep at smu2 (can be a numpy
                array, list or tuple).
            t_int (float): Integration time per data point (float), must be
                between 0.001 to 25 times the power line frequency
            delay (float): Settling delay before measurement.
            pulsed (float): Continuous or pulsed sweep.

        Returns:
            v_smu1 (list): Voltages measurement during the sweep in Volts at
                the first smu.
            i_smu1 (list): Currents measurement during the sweep in Amperes at
                the first smu.
            v_smu2 (list): Voltages measurement during the sweep in Volts at
                the second smu.
            i_smu2 (list): Currents measurement during the sweep in Amperes at
                the second smu.
        """

        # input checks
        self._check_smu(smu1)
        self._check_smu(smu2)

        assert len(smu1_sweeplist) == len(smu2_sweeplist)

        # set state to busy
        self.busy = True
        # Define lists containing results. If we abort early, we have something to return.
        v_smu1, i_smu1, v_smu2, i_smu2 = [], [], [], []

        if self.abort_event.is_set():
            self.busy = False
            return v_smu1, i_smu1, v_smu2, i_smu2

        # Setup smua/smub for sweep measurement.

        # setup smu1 and smu2 to sweep through lists on trigger
        # send sweep_list over in chunks if too long
        if len(smu1_sweeplist) > self.CHUNK_SIZE:
            self._write('mylist = {}')
            for num in smu1_sweeplist:
                self._write('table.insert(mylist, %s)' % num)
            smu1.trigger.source.listv('mylist')
        else:
            smu1.trigger.source.listv(smu1_sweeplist)

        if len(smu2_sweeplist) > self.CHUNK_SIZE:
            self._write('mylist = {}')
            for num in smu2_sweeplist:
                self._write('table.insert(mylist, %s)' % num)
            smu2.trigger.source.listv('mylist')
        else:
            smu2.trigger.source.listv(smu2_sweeplist)

        smu1.trigger.source.action = smu1.ENABLE
        smu2.trigger.source.action = smu2.ENABLE

        # CONFIGURE INTEGRATION TIME FOR EACH MEASUREMENT
        self.setIntegrationTime(smu1, t_int)
        self.setIntegrationTime(smu2, t_int)

        # CONFIGURE SETTLING TIME FOR GATE VOLTAGE, I-LIMIT, ETC...
        smu1.measure.delay = delay
        smu2.measure.delay = delay

        smu1.measure.autorangei = smu1.AUTORANGE_ON
        smu2.measure.autorangei = smu2.AUTORANGE_ON

        # smu1.trigger.source.limiti = 0.1
        # smu2.trigger.source.limiti = 0.1

        smu1.source.func = smu1.OUTPUT_DCVOLTS
        smu2.source.func = smu2.OUTPUT_DCVOLTS

        # 2-wire measurement (use SENSE_REMOTE for 4-wire)
        # smu1.sense = smu1.SENSE_LOCAL
        # smu2.sense = smu2.SENSE_LOCAL

        # CLEAR BUFFERS
        for smu in [smu1, smu2]:
            smu.nvbuffer1.clear()
            smu.nvbuffer2.clear()
            smu.nvbuffer1.clearcache()
            smu.nvbuffer2.clearcache()

        # display current values during measurement
        self.display.smua.measure.func = self.display.MEASURE_DCAMPS
        self.display.smub.measure.func = self.display.MEASURE_DCAMPS

        # SETUP TRIGGER ARM AND COUNTS
        # trigger count = number of data points in measurement
        # arm count = number of times the measurement is repeated (set to 1)

        npts = len(smu1_sweeplist)

        smu1.trigger.count = npts
        smu2.trigger.count = npts

        # SET THE MEASUREMENT TRIGGER ON BOTH SMU'S
        # Set measurement to trigger once a change in the gate value on
        # sweep smu is complete, i.e., a measurement will occur
        # after the voltage is stepped.
        # Both channels should be set to trigger on the sweep smu event
        # so the measurements occur at the same time.

        # enable smu
        smu1.trigger.measure.action = smu1.ENABLE
        smu2.trigger.measure.action = smu2.ENABLE

        # measure current and voltage on trigger, store in buffer of smu
        smu1.trigger.measure.iv(smu1.nvbuffer1, smu1.nvbuffer2)
        smu2.trigger.measure.iv(smu2.nvbuffer1, smu2.nvbuffer2)

        # initiate measure trigger when source is complete
        smu1.trigger.measure.stimulus = smu1.trigger.SOURCE_COMPLETE_EVENT_ID
        smu2.trigger.measure.stimulus = smu1.trigger.SOURCE_COMPLETE_EVENT_ID

        # SET THE ENDPULSE ACTION TO HOLD
        # Options are SOURCE_HOLD AND SOURCE_IDLE, hold maintains same voltage
        # throughout step in sweep (typical IV sweep behavior). idle will allow
        # pulsed IV sweeps.

        if pulsed:
            end_pulse_action = 0  # SOURCE_IDLE
        elif not pulsed:
            end_pulse_action = 1  # SOURCE_HOLD
        else:
            raise TypeError("'pulsed' must be of type 'bool'.")

        smu1.trigger.endpulse.action = end_pulse_action
        smu2.trigger.endpulse.action = end_pulse_action

        # SET THE ENDSWEEP ACTION TO HOLD IF NOT PULSED
        # Output voltage will be held after sweep is done!

        smu1.trigger.endsweep.action = end_pulse_action
        smu2.trigger.endsweep.action = end_pulse_action

        # SET THE EVENT TO TRIGGER THE SMU'S TO THE ARM LAYER
        # A typical measurement goes from idle -> arm -> trigger.
        # The 'trigger.event_id' option sets the transition arm -> trigger
        # to occur after sending *trg to the instrument.

        smu1.trigger.arm.stimulus = self.trigger.EVENT_ID

        # Prepare an event blender (blender #1) that triggers when
        # the smua enters the trigger layer or reaches the end of a
        # single trigger layer cycle.

        # triggers when either of the stimuli are true ('or enable')
        self.trigger.blender[1].orenable = True
        self.trigger.blender[1].stimulus[1] = smu1.trigger.ARMED_EVENT_ID
        self.trigger.blender[1].stimulus[2] = smu1.trigger.PULSE_COMPLETE_EVENT_ID

        # SET THE smu1 SOURCE STIMULUS TO BE EVENT BLENDER #1
        # A source measure cycle within the trigger layer will occur when
        # either the trigger layer is entered (termed 'armed event') for the
        # first time or a single cycle of the trigger layer is complete (termed
        # 'pulse complete event').

        smu1.trigger.source.stimulus = self.trigger.blender[1].EVENT_ID

        # PREPARE AN EVENT BLENDER (blender #2) THAT TRIGGERS WHEN BOTH SMU'S
        # HAVE COMPLETED A MEASUREMENT.
        # This is needed to prevent the next source measure cycle from occurring
        # before the measurement on both channels is complete.

        self.trigger.blender[2].orenable = False  # triggers when both stimuli are true
        self.trigger.blender[2].stimulus[1] = smu1.trigger.MEASURE_COMPLETE_EVENT_ID
        self.trigger.blender[2].stimulus[2] = smu2.trigger.MEASURE_COMPLETE_EVENT_ID

        # SET THE smu1 ENDPULSE STIMULUS TO BE EVENT BLENDER #2
        smu1.trigger.endpulse.stimulus = self.trigger.blender[2].EVENT_ID

        # TURN ON smu1 AND smu2
        smu1.source.output = smu1.OUTPUT_ON
        smu2.source.output = smu2.OUTPUT_ON

        # INITIATE MEASUREMENT
        # prepare SMUs to wait for trigger
        smu1.trigger.initiate()
        smu2.trigger.initiate()
        # send trigger
        self._write('*trg')

        # CHECK STATUS BUFFER FOR MEASUREMENT TO FINISH
        # Possible return values:
        # 6 = smua and smub sweeping
        # 4 = only smub sweeping
        # 2 = only smua sweeping
        # 0 = neither smu sweeping

        status = 0
        while status == 0:  # while loop that runs until the sweep begins
            status = self.status.operation.sweeping.condition
            time.sleep(0.1)

        while status > 0:  # while loop that runs until the sweep ends
            status = self.status.operation.sweeping.condition
            time.sleep(0.1)

        # EXTRACT DATA FROM SMU BUFFERS
        i_smu1 = self.readBuffer(smu1.nvbuffer1)
        v_smu1 = self.readBuffer(smu1.nvbuffer2)
        i_smu2 = self.readBuffer(smu2.nvbuffer1)
        v_smu2 = self.readBuffer(smu2.nvbuffer2)

        # CLEAR BUFFERS
        for smu in [smu1, smu2]:
            smu.nvbuffer1.clear()
            smu.nvbuffer2.clear()
            smu.nvbuffer1.clearcache()
            smu.nvbuffer2.clearcache()

        self.busy = False

        return v_smu1, i_smu1, v_smu2, i_smu2

# =============================================================================
# Define higher level control functions
# =============================================================================

    def transferMeasurement(self, smu_gate, smu_drain, vg_start, vg_stop,
                            vg_step, vd_list, t_int, delay, pulsed):
        """
        Records a transfer curve and saves the results in a TransistorSweepData
        instance.

        Args:
            smu_gate: SMU attached to gate electrode of FET for transfer
                measurement (keithley smu object).
            smu_drain: SMU attached to drain electrode of FET for transfer
                measurement (keithley smu object).
            vg_start (float): Start voltage of transfer sweep in Volts .
            vg_stop (float): End voltage of transfer sweep in Volts.
            vg_step (float): Voltage step size for transfer sweep in Volts.
            vd_list (list): List of drain voltage steps in Volts.
            t_int (float): Integration time in sec for every data point.
            delay (float): Settling time in sec before every measurement. Set
                to -1 for for automatic delay.
            pulsed (bool): True or False for pulsed or continuous measurements.

        Returns:
            Returns a TransistorSweepData object containing sweep data.
        """
        self.busy = True
        self.abort_event.clear()

        msg = ('Recording transfer curve with Vg from %sV to %sV, Vd = %s V. '
               % (vg_start, vg_stop, vd_list))
        logger.info(msg)

        # create array with gate voltage steps, always include a step >= VgStop
        step = np.sign(vg_stop - vg_start) * abs(vg_step)
        sweeplist_gate_fwd = np.arange(vg_start, vg_stop + step, step)
        sweeplist_gate_rvs = np.flip(sweeplist_gate_fwd, 0)
        sweeplist_gate = np.append(sweeplist_gate_fwd, sweeplist_gate_rvs)

        # create ResultTable instance
        params = {'sweep_type': 'transfer', 't_int': t_int, 'delay': delay,
                  'pulsed': pulsed}
        rt = TransistorSweepData(params=params)
        rt.append_column(sweeplist_gate, name='Gate voltage', unit='V')

        # record sweeps for every drain voltage step
        for vdrain in vd_list:

            # check for abort event
            if self.abort_event.is_set():
                self.reset()
                self.beeper.beep(0.3, 2400)
                return rt

            # create array with drain voltages
            if vdrain == 'trailing':
                sweeplist_drain = sweeplist_gate
            else:
                sweeplist_drain = np.full_like(sweeplist_gate, vdrain)

            # conduct sweep
            v_g, i_g, v_d, i_d = self.voltageSweepDualSMU(
                    smu_gate, smu_drain, sweeplist_gate, sweeplist_drain, t_int,
                    delay, pulsed
                    )

            if not self.abort_event.is_set():
                i_s = np.array(i_d) + np.array(i_g)
                rt.append_column(i_s, name='Source current (Vd = %s)' % vdrain, unit='A')
                rt.append_column(i_d, name='Drain current (Vd = %s)' % vdrain, unit='A')
                rt.append_column(i_g, name='Gate current (Vd = %s)' % vdrain, unit='A')

        self.reset()
        self.beeper.beep(0.3, 2400)

        self.busy = False
        return rt

    def outputMeasurement(self, smu_gate, smu_drain, vd_start, vd_stop, vd_step,
                          vg_list, t_int, delay, pulsed):
        """
        Records an output curve and saves the results in a TransistorSweepData
        instance.

        Args:
            smu_gate: SMU attached to gate electrode of FET for transfer
                measurement (keithley smu object).
            smu_drain: SMU attached to drain electrode of FET for transfer
                measurement (keithley smu object).
            vd_start (float): Start voltage of output sweep in Volts .
            vd_stop (float): End voltage of output sweep in Volts.
            vd_step (float): Voltage step size for output sweep in Volts.
            vg_list (list): List of gate voltage steps in Volts.
            t_int (float): Integration time in sec for every data point.
            delay (float): Settling time in sec before every measurement. Set
                to -1 for for automatic delay.
            pulsed (bool): True or False for pulsed or continuous measurements.

        Returns:
            Returns a TransistorSweepData object containing sweep data.
        """

        self.busy = True
        self.abort_event.clear()
        msg = ('Recording output curve with Vd from %sV to %sV, Vg = %s V. '
               % (vd_start, vd_stop, vg_list))
        logger.info(msg)

        # create array with drain voltage steps, always include a step >= VgStop
        step = np.sign(vd_stop - vd_start) * abs(vd_step)
        sweeplist_drain_fwd = np.arange(vd_start, vd_stop + step, step)
        sweeplist_drain_rvs = np.flip(sweeplist_drain_fwd, 0)
        sweeplist_drain = np.append(sweeplist_drain_fwd, sweeplist_drain_rvs)

        # create ResultTable instance
        params = {'sweep_type': 'output', 't_int': t_int, 'delay': delay,
                  'pulsed': pulsed}
        rt = TransistorSweepData(params=params)
        rt.append_column(sweeplist_drain, name='Drain voltage', unit='V')

        for vgate in vg_list:
            if self.abort_event.is_set():
                self.reset()
                self.beeper.beep(0.3, 2400)
                return rt

            # create array with gate voltages
            sweeplist_gate = np.full_like(sweeplist_drain, vgate)

            # conduct forward sweep
            v_d, i_d, v_g, i_g = self.voltageSweepDualSMU(
                    smu_drain, smu_gate, sweeplist_drain, sweeplist_gate, t_int,
                    delay, pulsed
                    )

            if not self.abort_event.is_set():
                i_s = np.array(i_d) + np.array(i_g)
                rt.append_column(i_s, name='Source current (Vd = %s)' % vgate, unit='A')
                rt.append_column(i_d, name='Drain current (Vd = %s)' % vgate, unit='A')
                rt.append_column(i_g, name='Gate current (Vd = %s)' % vgate, unit='A')

        self.reset()
        self.beeper.beep(0.3, 2400)

        self.busy = False
        return rt

    def playChord(self, direction='up'):
        """Plays a chord on the Keithley.

        Args:
            direction (str): 'up' or 'done' for upward or downward chord
        """
        if direction is 'up':
            self.beeper.beep(0.3, 1046.5)
            self.beeper.beep(0.3, 1318.5)
            self.beeper.beep(0.3, 1568)

        elif direction is 'down':
            self.beeper.beep(0.3, 1568)
            self.beeper.beep(0.3, 1318.5)
            self.beeper.beep(0.3, 1046.5)
        else:
            self.beeper.beep(0.2, 1046.5)
            self.beeper.beep(0.1, 1046.5)


class Keithley2600Factory(object):

    _instances = {}
    SMU_LIST = Keithley2600.SMU_LIST

    def __new__(cls, *args, **kwargs):
        """
        Create new instance for a new visa_address, otherwise return existing instance.
        """
        if args[0] in cls._instances:
            logger.debug('Returning existing instance with address %s.' % args[0])

            return cls._instances[args[0]]
        else:
            logger.debug('Creating new instance with address %s.' % args[0])
            instance = Keithley2600(*args, **kwargs)
            cls._instances[args[0]] = instance

            return instance
