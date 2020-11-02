# -*- coding: utf-8 -*-
#
# Copyright © keithley2600 Project Contributors
# Licensed under the terms of the MIT License
# (see keithley2600/__init__.py for details)

"""
Core driver with the low level functions.
"""

# system imports
import re
import logging
import threading
import numpy as np
import time
from threading import RLock
from xdrlib import Error as XDRError
import pyvisa

# local import
from keithley2600.keithley_doc import (
    CONSTANTS,
    FUNCTIONS,
    PROPERTIES,
    CLASSES,
    PROPERTY_LISTS,
    ALL_CMDS,
)
from keithley2600.result_table import FETResultTable


logger = logging.getLogger(__name__)


def log_to_screen(level=logging.DEBUG):
    log_to_stream(None, level)  # sys.stderr by default


def log_to_stream(stream_output, level=logging.DEBUG):
    logger.setLevel(level)
    ch = logging.StreamHandler(stream_output)
    ch.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch.setFormatter(formatter)

    logger.addHandler(ch)


def removeprefix(self: str, prefix: str) -> str:
    """
    Removes the given prefix from a string. Only the first instance of the prefix is
    removed. The original string is returned if it does not start with the given prefix.

    This follows the Python 3.9 implementation of ``str.removeprefix``.

    :param self: Original string.
    :param prefix: Prefix to remove.
    :returns: String without prefix.
    """
    if self.startswith(prefix):
        return self[len(prefix) :]
    else:
        return self[:]


class MagicPropertyList:
    """Mimics a Keithley TSP property list

    Class which mimics a Keithley TSP property list and can be dynamically
    created. It forwards all calls to the :func:`_read` method of the parent
    class and assignments to the :func:`_write` method. Arbitrary values can be
    assigned, as long as :func:`_write` can handle them.

    This class is designed to look like a Keithley TSP "attribute" list,
    forward function calls to the Keithley, and return the results.

    """

    def __init__(self, name, parent):
        if not isinstance(name, str):
            raise ValueError("First argument must be of type str.")
        self._name = name
        self._parent = parent

    def __getitem__(self, i):
        """Gets i-th item: query item from parent class

        :param int i: An integer item number

        :returns: Result from _query call of parent class.
        """
        new_name = "%s[%s]" % (self._name, i)
        return self._parent._query(new_name)

    def __setitem__(self, i, value):
        """Sets i-th item: set item at parent class

        :param int i: An integer item number
        :param value: An input object that can be accepted by parent class.

        """
        value = self._parent._convert_input(value)
        new_name = "%s[%s] = %s" % (self._name, i, value)
        self._parent._write(new_name)

    def __iter__(self):
        return self

    def getdoc(self):
        """Prevent pydoc from trying to document this class. This could conflict with
        on-demand creation of attributes."""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}({self._name})>"


class MagicFunction:
    """Mimics a Keithley TSP function

    Class which mimics a function and can be dynamically created. It forwards
    all calls to the :func:`_query` method of the parent class and returns the
    result from :func:`_query`. Calls accept arbitrary arguments, as long as
    :func:`_query` can handle them.

    This class is designed to look like a Keithley TSP function, forward
    function calls to the Keithley, and return the results.
    """

    def __init__(self, name, parent):
        if not isinstance(name, str):
            raise ValueError("First argument must be of type str.")
        self._name = name
        self._parent = parent

    def __call__(self, *args, **kwargs):
        """Pass on calls to :func:`parent._write`, store result in variable.
        Querying results from function calls directly may result in
        a VisaIOError if the function does not return anything."""

        # convert incompatible arguments, return all arguments as tuple
        args = tuple(self._parent._convert_input(a) for a in args)
        # remove outside brackets and all quotation marks
        args_string = str(args).strip("(),").replace("'", "")

        # pass on a string representation of the function call to self._parent._query
        return self._parent._query("%s(%s)" % (self._name, args_string))

    def __repr__(self):
        return f"<{self.__class__.__name__}({self._name})>"


class MagicClass:
    """Mimics a TSP command group

    Class which dynamically creates new attributes on access. These can be
    functions, properties, or other classes.

    Attribute setters and getters are forwarded to :func:`_write` and
    :func:`_query` functions from the parent class. New functions are created
    as instances of :class:`MagicFunction`, new classes are created as
    instances of :class:`MagicClass`.

    MagicClass is designed to mimic a Keithley TSP command group with
    functions, attributes, and subordinate command groups.

    :Examples:

        >>> inst = MagicClass('keithley')
        >>> inst.reset()  # Dynamically creates a new attribute 'reset' as an
        ...               # instance of MagicFunction, then calls it.
        >>> inst.beeper  # Dynamically creates new attribute 'beeper' and sets
        ...              # it to a new MagicClass instance.
        >>> inst.beeper.enable  # Fakes the property 'enable' of 'beeper'
        ...                     # with _write as setter and _query as getter.

    """

    _name = ""
    _parent = None

    def __init__(self, name, parent=None):
        self._name = name
        self.__dict = {}
        if parent is not None:
            self._parent = parent

    def __getattr__(self, attr_name):
        """Custom getter

        Get attributes as usual if they exist. Otherwise, fall back to
        :func:`__get_global_handler`.
        """

        if not self.__dict:
            self._populate_dict()

        try:
            try:
                # check if attribute already exists. return attr if yes.
                return super().__getattr__(attr_name)
            except AttributeError:
                # check if key already exists. return value if yes.
                return self.__dict[attr_name]
        except KeyError:
            raise AttributeError(f"{self} has no attribute '{attr_name}'")

    def _populate_dict(self):

        var_name, var_type = self._query(f"next({self._name}, nil)")

        if var_type.startswith("function"):
            self.__dict[var_name] = MagicFunction(var_name, self)
        elif var_type.startswith("table"):
            self.__dict[var_name] = MagicClass(var_name, self)

    def __setattr__(self, attr_name, value):
        """Custom setter

        Forward setting commands to `self._write` for expected Keithley TSP
        attributes. Otherwise use default setter.

        :param str attr_name: Attribute name.
        :param value: Value to set.

        :raises: :class:`ValueError` if trying to write a value to read-only
            Keithley attributes.
        """
        if attr_name in PROPERTIES:
            value = self._convert_input(value)
            self._write("%s.%s = %s" % (self._name, attr_name, value))
        elif attr_name in CONSTANTS:
            raise ValueError("%s.%s is read-only." % (self._name, attr_name))
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
        new_name = "%s[%s]" % (self._name, i)
        new_class = MagicClass(new_name, parent=self)
        return new_class

    def __iter__(self):
        return self

    def __dir__(self):
        return list(self.__dict.keys()) + super().__dir__()

    def __repr__(self):
        return f"<{self.__class__.__name__}({self._name})>"


class KeithleyIOError(Exception):
    """Raised when no Keithley instrument is connected."""

    pass


class KeithleyError(Exception):
    """Raised for error messages from the Keithley itself."""

    pass


class Keithley2600Base(MagicClass):
    """Keithley2600 driver

    Keithley driver for base functionality. It replicates the functionality and
    syntax from the Keithley TSP commands, which have a syntax similar to
    python. Attributes are created on-access if they correspond to Keithley TSP
    type commands.

    :param str visa_address: Visa address of the instrument.
    :param str visa_library: Path to visa library. Defaults to "@py" for pyvisa-py
        but another IVI library may be appropriate (NI-VISA, Keysight VISA, R&S VISA,
        tekVISA etc.). If an empty string is given, an IVI library will be used if
        installed and pyvisa-py otherwise.
    :param bool raise_keithley_errors: If ``True``, all Keithley errors will be
        raised as Python errors instead of being ignored. This causes
        significant communication overhead because the Keithley's error queue
        is read after each command. Defaults to ``False``.
    :param kwargs: Keyword arguments passed on to the visa connection, for
        instance baude-rate or timeout. If not given, reasonable defaults will
        be used.

    :cvar connection: Attribute holding a reference to the actual connection.
    :cvar bool connected: ``True`` if connected to an instrument, ``False``
        otherwise.
    :cvar bool busy: ``True`` if a measurement is running, ``False`` otherwise.
    :cvar int CHUNK_SIZE: Maximum length of lists which can be sent to the
        Keithley. Longer lists will be transferred in chunks.

    .. note::

        See the Keithley 2600 reference manual for all available commands and
        arguments. Almost all Keithley TSP commands can be used with this
        driver. Not supported are:

         - ``lan.trigger[N].connected``: conflicts with the connected attribute
         - ``io.output()``: conflicts with ``smuX.source.output``
         - All Keithley IV sweep commands. We implement our own in
           :class:`Keithley2600`.

    :Examples:

        >>> keithley = Keithley2600Base('TCPIP0::192.168.2.121::INSTR')
        >>> keithley.smua.measure.v()  # measures voltage at smuA
        >>> keithley.smua.source.levelv = -40  # applies -40V to smuA

    """

    connection = None
    connected = False
    busy = False

    CHUNK_SIZE = 50

    SMU_LIST = []

    _lock = RLock()

    def __init__(
        self, visa_address, visa_library="@py", raise_keithley_errors=False, **kwargs
    ):

        MagicClass.__init__(self, name="", parent=self)
        self._name = ""

        self.abort_event = threading.Event()

        self.visa_address = visa_address
        self.visa_library = visa_library
        self._connection_kwargs = kwargs

        self.raise_keithley_errors = raise_keithley_errors

        # open visa resource manager with selected library / backend
        self.rm = pyvisa.ResourceManager(self.visa_library)
        # connect to keithley
        self.connect(**kwargs)

    def __repr__(self):
        return "<%s(%s)>" % (type(self).__name__, self.visa_address)

    # =============================================================================
    # Connect to keithley
    # =============================================================================

    def connect(self, **kwargs):
        """
        Connects to Keithley.

        :param kwargs: Keyword arguments for Visa connection.

        """
        kwargs = kwargs or self._connection_kwargs  # use specified or remembered kwargs
        try:
            self.connection = self.rm.open_resource(self.visa_address, **kwargs)
            self.connection.read_termination = "\n"
            self.connected = True
            logger.debug("Connected to Keithley at %s." % self.visa_address)
        except ValueError:
            self.connection = None
            self.connected = False
            raise
        except ConnectionError:
            logger.info(
                "Connection error. Please check that "
                + "no other program is connected."
            )
            self.connection = None
            self.connected = False
        except AttributeError:
            logger.info("Invalid VISA address %s." % self.visa_address)
            self.connection = None
            self.connected = False
        except Exception:
            logger.info("Could not connect to Keithley at %s." % self.visa_address)
            self.connection = None
            self.connected = False
        else:
            # get list of SMUs

            for suffix in ("a", "b", "c", "d"):
                smu_name = f"smu{suffix}"
                if self._query(smu_name) is not None:
                    self.SMU_LIST.append(smu_name)

    def disconnect(self):
        """
        Disconnects from Keithley.
        """
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                self.connected = False
                del self.connection
                logger.debug("Disconnected from Keithley at %s." % self.visa_address)
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

        with self._lock:
            logger.debug("write: %s" % value)

            if self.connection:

                if self.raise_keithley_errors and "errorqueue" not in value:
                    self.errorqueue.clear()

                self.connection.write(value)

                if self.raise_keithley_errors and "errorqueue" not in value:
                    err = self.errorqueue.next()
                    if err[0] != 0:
                        raise KeithleyError(
                            "Error during command '{0}': {1}".format(value, err[1])
                        )
            else:
                raise KeithleyIOError(
                    "No connection to keithley present. Try to call 'connect'."
                )

    def _query(self, value):
        """
        Queries and expects response from Keithley. Input must be a string.
        """

        # only check for error when the query is not fetching the error queue
        check_for_errors = self.raise_keithley_errors and "errorqueue" not in value

        with self._lock:
            logger.debug("write: print(%s)" % value)
            if self.connection:

                if check_for_errors:
                    # clear any previous errors
                    self.errorqueue.clear()

                try:
                    r = self.connection.query("print(%s)" % value)
                    logger.debug("read: %s" % r)
                except XDRError:
                    r = "nil"
                    logger.debug("read failed: unpack-error")

                if check_for_errors:
                    # read error queue to check for new errors
                    err = self.errorqueue.next()
                    if err[0] != 0:
                        err_msg = err[1].replace("TSP Runtime error at line 1: ", "")
                        raise KeithleyError(
                            "Error during command '{0}': {1}".format(value, err_msg)
                        )

                return self._parse_response(r)
            else:
                raise KeithleyIOError(
                    "No connection to keithley present. Try to call 'connect'."
                )

    @staticmethod
    def _parse_single_response(string):

        # Dictionary to convert from Keithley TSP to Python types.
        # Note that emtpy strings are converted to `None`. This is necessary
        # since `self.connection.query('print(func())')` returns an empty
        # string if the TSP function `func()` returns 'nil'.
        conversion_dict = {"true": True, "false": False, "nil": None, "": None}

        try:
            r = float(string)
            if r.is_integer():
                r = int(r)
        except ValueError:
            if string in conversion_dict.keys():
                r = conversion_dict[string]
            else:
                r = string

        return r

    def _parse_response(self, string):

        string_list = string.split("\t")

        converted_tuple = tuple(self._parse_single_response(s) for s in string_list)

        if len(converted_tuple) == 1:
            return converted_tuple[0]
        else:
            return converted_tuple

    def _convert_input(self, value):
        """Convert bool to lower case string and list / tuples to comma
        delimited string enclosed by curly brackets."""
        if isinstance(value, bool):
            # convert bool True to string 'true'
            return str(value).lower()
        elif isinstance(value, MagicClass):
            # convert keithley object to string with its name
            return value._name
        elif hasattr(value, "__iter__"):
            # convert some iterables to a TSP type list '{1,2,3,4}'
            return "{" + ", ".join([str(v) for v in value]) + "}"
        else:
            return value


class Keithley2600(Keithley2600Base):
    """Keithley2600 driver with high level functionality

    Keithley driver with access to base functions and higher level functions
    such as IV measurements, transfer and output curves, etc. Inherits from
    :class:`Keithley2600Base`. Base commands replicate the functionality and
    syntax of Keithley TSP functions.

    :param str visa_address: Visa address of the instrument.
    :param str visa_library: Path to visa library. Defaults to "@py" for pyvisa-py
        but another IVI library may be appropriate (NI-VISA, Keysight VISA, R&S VISA,
        tekVISA etc.). If an empty string is given, an IVI library will be used if
        installed and pyvisa-py otherwise.
    :param bool raise_keithley_errors: If ``True``, all Keithley errors will be
        raised as Python errors instead of being ignored. This causes
        significant communication overhead because the Keithley's error queue
        is read after each command. Defaults to ``False``.
    :param kwargs: Keyword arguments passed on to the visa connection, for
        instance baude-rate or timeout. If not given, reasonable defaults will
        be used.

    :cvar connection: Attribute holding a reference to the actual connection.
    :cvar bool connected: ``True`` if connected to an instrument, ``False``
        otherwise.
    :cvar bool busy: ``True`` if a measurement is running, ``False`` otherwise.

    :Examples:

        *Base commands from Keithley TSP*:

        >>> k = Keithley2600('TCPIP0::192.168.2.121::INSTR')
        >>> volts = k.smua.measure.v()  # measures and returns the smuA voltage
        >>> k.smua.source.levelv = -40  # sets source level of smuA
        >>> k.smua.nvbuffer1.clear()  # clears nvbuffer1 of smuA

        *New mid-level commands*:

        >>> data = k.readBuffer(k.smua.nvbuffer1)
        >>> errs = k.readErrorQueue()
        >>> k.setIntegrationTime(k.smua, 0.001) # in sec

        >>> k.applyVoltage(k.smua, -60) # applies -60V to smuA
        >>> k.applyCurrent(k.smub, 0.1) # sources 0.1A from smuB
        >>> k.rampToVoltage(k.smua, 10, delay=0.1, step_size=1)

        >>> # voltage sweeps, single and dual SMU
        >>> k.voltageSweepSingleSMU(smu=k.smua, smu_sweeplist=range(0, 61),
        ...                         t_int=0.1, delay=-1, pulsed=False)
        >>> k.voltageSweepDualSMU(smu1=k.smua, smu2=k.smub,
        ...                       smu1_sweeplist=range(0, 61),
        ...                       smu2_sweeplist=range(0, 61),
        ...                       t_int=0.1, delay=-1, pulsed=False)

        *New high-level commands*:

        >>> data1 = k.outputMeasurement(...) # records output curve
        >>> data2 = k.transferMeasurement(...) # records transfer curve

    """

    def __init__(
        self, visa_address, visa_library="@py", raise_keithley_errors=False, **kwargs
    ):
        Keithley2600Base.__init__(
            self,
            visa_address,
            visa_library,
            raise_keithley_errors=raise_keithley_errors,
            **kwargs,
        )

    def __repr__(self):
        return "<%s(%s)>" % (type(self).__name__, self.visa_address)

    def _check_smu(self, smu):
        """
        Check if selected smu is indeed present.

        :param smu: A keithley smu instance.
        """

        if self._get_smu_name(smu) not in self.SMU_LIST:
            raise RuntimeError("The specified SMU does not exist.")

    @staticmethod
    def _get_smu_name(smu):
        return smu._name.split(".")[-1]

    # =============================================================================
    # Define lower level control functions
    # =============================================================================

    def readErrorQueue(self):
        """
        Returns all entries from the Keithley error queue and clears the queue.

        :returns: List of errors from the Keithley error queue. Each entry is a
            tuple ``(error_code, message, severity, error_node)``. If the queue
            is empty, an empty list is returned.
        :rtype: list
        """

        error_list = []

        err = self.errorqueue.next()
        while err[0] != 0:
            error_list += err
            err = self.errorqueue.next()

        return error_list

    @staticmethod
    def readBuffer(buffer):
        """
        Reads buffer values and returns them as a list. This can be done more
        quickly by calling :attr:`buffer.readings` but such a call may fail due
        to I/O limitations of the keithley if the returned list is too long.

        :param buffer: A keithley buffer instance.

        :returns: A list with buffer readings.
        :rtype: list
        """
        list_out = []
        for i in range(0, int(buffer.n)):
            list_out.append(buffer.readings[i + 1])

        return list_out

    def setIntegrationTime(self, smu, t_int):
        """
        Sets the integration time of SMU for measurements in sec.

        :param smu: A keithley smu instance.
        :param float t_int: Integration time in sec. Value must be between
            1/1000 and 25 power line cycles (50Hz or 60 Hz).
        :raises: :class:`ValueError` for too short or too long integration
            times.
        """

        self._check_smu(smu)

        # determine number of power-line-cycles used for integration
        freq = self.localnode.linefreq
        nplc = t_int * freq

        if nplc < 0.001 or nplc > 25:
            raise ValueError(
                "Integration time must be between 0.001 and 25 "
                + "power line cycles of 1/(%s Hz)." % freq
            )
        smu.measure.nplc = nplc

    def applyVoltage(self, smu, voltage):
        """
        Turns on the specified SMU and applies a voltage.

        :param smu: A keithley smu instance.
        :param float voltage: Voltage to apply in Volts.
        """

        self._check_smu(smu)

        smu.source.levelv = voltage
        smu.source.func = smu.OUTPUT_DCVOLTS
        smu.source.output = smu.OUTPUT_ON

    def applyCurrent(self, smu, curr):
        """
        Turns on the specified SMU and sources a current.

        :param smu: A keithley smu instance.
        :param float curr: Current to apply in Ampere.
        """
        self._check_smu(smu)

        smu.source.leveli = curr
        smu.source.func = smu.OUTPUT_DCAMPS
        smu.source.output = smu.OUTPUT_ON

    def measureVoltage(self, smu):
        """
        Measures a voltage at the specified SMU.

        :param smu: A keithley smu instance.

        :returns: Measured voltage in Volts.
        :rtype: float
        """

        self._check_smu(smu)

        return smu.measure.v()

    def measureCurrent(self, smu):
        """
        Measures a current at the specified SMU.

        :param smu: A keithley smu instance.

        :returns: Measured current in Ampere.
        :rtype: float
        """
        self._check_smu(smu)

        return smu.measure.i()

    def rampToVoltage(self, smu, target_volt, delay=0.1, step_size=1):
        """
        Ramps up the voltage of the specified SMU. Beeps when done.

        :param smu: A keithley smu instance.
        :param float target_volt: Target voltage in Volts.
        :param float step_size: Size of the voltage steps in Volts.
        :param float delay: Delay between steps in sec.
        """

        self._check_smu(smu)

        smu.source.output = smu.OUTPUT_ON

        # get current voltage
        vcurr = smu.source.levelv
        if vcurr == target_volt:
            return

        step = np.sign(target_volt - vcurr) * abs(step_size)

        for v in np.arange(vcurr, target_volt, step):
            smu.source.levelv = v
            smu.measure.v()
            time.sleep(delay)

        smu.source.levelv = target_volt
        logger.info("Gate voltage set to Vg = %s V." % round(target_volt))

        self.beeper.beep(0.3, 2400)

    def voltageSweepSingleSMU(self, smu, smu_sweeplist, t_int, delay, pulsed):
        """
        Sweeps the voltage through the specified list of steps at the given
        SMU. Measures and returns the current and voltage during the sweep.

        :param smu: A keithley smu instance.
        :param smu_sweeplist: Voltages to sweep through (can be a numpy array,
             list, tuple or any other iterable with numbers).
        :param float t_int: Integration time per data point. Must be between
            0.001 to 25 times the power line frequency (50Hz or 60Hz).
        :param float delay: Settling delay before each measurement. A value of
            -1 automatically starts a measurement once the current is stable.
        :param bool pulsed: Select pulsed or continuous sweep. In a pulsed
            sweep, the voltage is always reset to zero between data points.

        :returns: Lists of voltages and currents measured during the sweep (in
            Volt and Ampere, respectively): ``(v_smu, i_smu)``.
        :rtype: (list, list)
        """

        # input checks
        self._check_smu(smu)

        # set state to busy
        self.busy = True
        # Define lists containing results.
        # If we abort early, we have something to return.
        v_smu, i_smu = [], []

        if self.abort_event.is_set():
            self.busy = False
            return v_smu, i_smu

        # setup smu to sweep through list on trigger
        # send sweep_list over in chunks if too long
        if len(smu_sweeplist) > self.CHUNK_SIZE:
            self._write("mylist = {}")
            for num in smu_sweeplist:
                self._write("table.insert(mylist, %s)" % num)
            smu.trigger.source.listv("mylist")
        else:
            smu.trigger.source.listv(smu_sweeplist)

        smu.trigger.source.action = smu.ENABLE

        # CONFIGURE INTEGRATION TIME FOR EACH MEASUREMENT
        self.setIntegrationTime(smu, t_int)

        # CONFIGURE SETTLING TIME FOR GATE VOLTAGE, I-LIMIT, ETC...
        smu.measure.delay = delay

        # enable autorange if not in high capacitance mode
        if smu.source.highc == smu.DISABLE:
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
        smu_name = self._get_smu_name(smu)
        getattr(self.display, smu_name).measure.func = self.display.MEASURE_DCAMPS

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
        self._write("*trg")

        # CHECK STATUS BUFFER FOR MEASUREMENT TO FINISH
        # Possible return values:
        # 6 = smua and smub sweeping
        # 4 = only smub sweeping
        # 2 = only smua sweeping
        # 0 = neither smu sweeping

        # while loop that runs until the sweep begins
        while self.status.operation.sweeping.condition == 0:
            time.sleep(0.1)

        # while loop that runs until the sweep ends
        while self.status.operation.sweeping.condition > 0:
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

    def voltageSweepDualSMU(
        self, smu1, smu2, smu1_sweeplist, smu2_sweeplist, t_int, delay, pulsed
    ):
        """
        Sweeps voltages at two SMUs. Measures and returns current and voltage
        during sweep.

        :param smu1: 1st keithley smu instance to be swept.
        :param smu2: 2nd keithley smu instance to be swept.
        :param smu1_sweeplist: Voltages to sweep at ``smu1`` (can be a numpy
             array, list, tuple or any other iterable with numbers).
        :param smu2_sweeplist: Voltages to sweep at ``smu2`` (can be a numpy
             array, list, tuple or any other iterable with numbers).
        :param float t_int: Integration time per data point. Must be
            between 0.001 to 25 times the power line frequency (50Hz or 60Hz).
        :param float delay: Settling delay before each measurement. A value of
            -1 automatically starts a measurement once the current is stable.
        :param bool pulsed: Select pulsed or continuous sweep. In a pulsed
            sweep, the voltage is always reset to zero between data points.

        :returns: Lists of voltages and currents measured during the sweep (in
            Volt and Ampere, respectively): ``(v_smu1, i_smu1, v_smu2,
            i_smu2)``.
        :rtype: (list, list, list, list)
        """

        # input checks
        self._check_smu(smu1)
        self._check_smu(smu2)

        if len(smu1_sweeplist) != len(smu2_sweeplist):
            raise ValueError("Sweep lists must have equal lengths")

        # set state to busy
        self.busy = True
        # Define lists containing results.
        # If we abort early, we have something to return.
        v_smu1, i_smu1, v_smu2, i_smu2 = [], [], [], []

        if self.abort_event.is_set():
            self.busy = False
            return v_smu1, i_smu1, v_smu2, i_smu2

        # Setup smua/smub for sweep measurement.

        # setup smu1 and smu2 to sweep through lists on trigger
        # send sweep_list over in chunks if too long
        if len(smu1_sweeplist) > self.CHUNK_SIZE:
            self._write("mylist = {}")
            for num in smu1_sweeplist:
                self._write("table.insert(mylist, %s)" % num)
            smu1.trigger.source.listv("mylist")
        else:
            smu1.trigger.source.listv(smu1_sweeplist)

        if len(smu2_sweeplist) > self.CHUNK_SIZE:
            self._write("mylist = {}")
            for num in smu2_sweeplist:
                self._write("table.insert(mylist, %s)" % num)
            smu2.trigger.source.listv("mylist")
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

        # enable autorange if not in high capacitance mode
        if smu1.source.highc == smu1.DISABLE:
            smu1.measure.autorangei = smu1.AUTORANGE_ON
        if smu2.source.highc == smu2.DISABLE:
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
        for smu in (smu1, smu2):
            smu_name = self._get_smu_name(smu)
            getattr(self.display, smu_name).measure.func = self.display.MEASURE_DCAMPS

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
        self._write("*trg")

        # CHECK STATUS BUFFER FOR MEASUREMENT TO FINISH
        # Possible return values:
        # 6 = smua and smub sweeping
        # 4 = only smub sweeping
        # 2 = only smua sweeping
        # 0 = neither smu sweeping

        # while loop that runs until the sweep begins
        while self.status.operation.sweeping.condition == 0:
            time.sleep(0.1)

        # while loop that runs until the sweep ends
        while self.status.operation.sweeping.condition > 0:
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

    def transferMeasurement(
        self,
        smu_gate,
        smu_drain,
        vg_start,
        vg_stop,
        vg_step,
        vd_list,
        t_int,
        delay,
        pulsed,
    ):
        """
        Records a transfer curve with forward and reverse sweeps and returns
        the results in a :class:`sweep_data.TransistorSweepData` instance.


        :param smu_gate: Keithley smu attached to gate electrode.
        :param smu_drain: Keithley smu attached to drain electrode.
        :param float vg_start: Start voltage of transfer sweep in Volt.
        :param float vg_stop: End voltage of transfer sweep in Volt.
        :param float vg_step: Voltage step size for transfer sweep in Volt.
        :param vd_list: List of drain voltage steps in Volt. Can be a numpy
             array, list, tuple, range / xrange.
        :param float t_int: Integration time per data point. Must be
            between 0.001 to 25 times the power line frequency (50Hz or 60Hz).
        :param float delay: Settling delay before each measurement. A value of
            -1 automatically starts a measurement once the current is stable.
        :param bool pulsed: Select pulsed or continuous sweep. In a pulsed
            sweep, the voltage is always reset to zero between data points.

        :returns: Transfer curve data.
        :rtype: :class:`sweep_data.TransistorSweepData`

        """
        self.busy = True
        self.abort_event.clear()

        msg = "Recording transfer curve with Vg from %sV to %sV, Vd = %s V. " % (
            vg_start,
            vg_stop,
            vd_list,
        )
        logger.info(msg)

        # create array with gate voltage steps, always include a step >= VgStop
        step = np.sign(vg_stop - vg_start) * abs(vg_step)
        sweeplist_gate_fwd = np.arange(vg_start, vg_stop + step, step)
        sweeplist_gate_rvs = np.flip(sweeplist_gate_fwd, 0)
        sweeplist_gate = np.append(sweeplist_gate_fwd, sweeplist_gate_rvs)

        # create ResultTable instance
        params = {
            "sweep_type": "transfer",
            "time": time.time(),
            "time_str": time.strftime("%d/%m/%Y %H:%M"),
            "t_int": t_int,
            "delay": delay,
            "pulsed": pulsed,
        }
        rt = FETResultTable(params=params)
        rt.append_column(sweeplist_gate, name="Gate voltage", unit="V")

        # record sweeps for every drain voltage step
        for vdrain in vd_list:

            # check for abort event
            if self.abort_event.is_set():
                self.reset()
                self.beeper.beep(0.3, 2400)
                return rt

            # create array with drain voltages
            if vdrain == "trailing":
                sweeplist_drain = sweeplist_gate
            else:
                sweeplist_drain = np.full_like(sweeplist_gate, vdrain)

            # conduct sweep
            v_g, i_g, v_d, i_d = self.voltageSweepDualSMU(
                smu_gate,
                smu_drain,
                sweeplist_gate,
                sweeplist_drain,
                t_int,
                delay,
                pulsed,
            )

            if not self.abort_event.is_set():
                i_s = np.array(i_d) + np.array(i_g)
                rt.append_column(
                    i_s, name="Source current (Vd = %s)" % vdrain, unit="A"
                )
                rt.append_column(i_d, name="Drain current (Vd = %s)" % vdrain, unit="A")
                rt.append_column(i_g, name="Gate current (Vd = %s)" % vdrain, unit="A")

        self.reset()
        self.beeper.beep(0.3, 2400)

        self.busy = False
        return rt

    def outputMeasurement(
        self,
        smu_gate,
        smu_drain,
        vd_start,
        vd_stop,
        vd_step,
        vg_list,
        t_int,
        delay,
        pulsed,
    ):
        """
        Records an output curve with forward and reverse sweeps and returns the
        results in a :class:`sweep_data.TransistorSweepData` instance.

        :param smu_gate: Keithley smu attached to gate electrode.
        :param smu_drain: Keithley smu attached to drain electrode.
        :param float vd_start: Start voltage of output sweep in Volt.
        :param float vd_stop: End voltage of output sweep in Volt.
        :param float vd_step: Voltage step size for output sweep in Volt.
        :param vg_list: List of gate voltage steps in Volt. Can be a numpy
             array, list, tuple, range / xrange.
        :param float t_int: Integration time per data point. Must be
            between 0.001 to 25 times the power line frequency (50Hz or 60Hz).
        :param float delay: Settling delay before each measurement. A value of
            -1 automatically starts a measurement once the current is stable.
        :param bool pulsed: Select pulsed or continuous sweep. In a pulsed
            sweep, the voltage is always reset to zero between data points.

        :returns: Output curve data.
        :rtype: :class:`sweep_data.TransistorSweepData`

        """

        self.busy = True
        self.abort_event.clear()
        msg = "Recording output curve with Vd from %sV to %sV, Vg = %s V. " % (
            vd_start,
            vd_stop,
            vg_list,
        )
        logger.info(msg)

        # create array with drain voltage steps, always include a step >= VgStop
        step = np.sign(vd_stop - vd_start) * abs(vd_step)
        sweeplist_drain_fwd = np.arange(vd_start, vd_stop + step, step)
        sweeplist_drain_rvs = np.flip(sweeplist_drain_fwd, 0)
        sweeplist_drain = np.append(sweeplist_drain_fwd, sweeplist_drain_rvs)

        # create ResultTable instance
        params = {
            "sweep_type": "output",
            "time": time.time(),
            "time_str": time.strftime("%d/%m/%Y %H:%M"),
            "t_int": t_int,
            "delay": delay,
            "pulsed": pulsed,
        }
        rt = FETResultTable(params=params)
        rt.append_column(sweeplist_drain, name="Drain voltage", unit="V")

        for vgate in vg_list:
            if self.abort_event.is_set():
                self.reset()
                self.beeper.beep(0.3, 2400)
                return rt

            # create array with gate voltages
            sweeplist_gate = np.full_like(sweeplist_drain, vgate)

            # conduct forward sweep
            v_d, i_d, v_g, i_g = self.voltageSweepDualSMU(
                smu_drain,
                smu_gate,
                sweeplist_drain,
                sweeplist_gate,
                t_int,
                delay,
                pulsed,
            )

            if not self.abort_event.is_set():
                i_s = np.array(i_d) + np.array(i_g)
                rt.append_column(i_s, name="Source current (Vg = %s)" % vgate, unit="A")
                rt.append_column(i_d, name="Drain current (Vg = %s)" % vgate, unit="A")
                rt.append_column(i_g, name="Gate current (Vg = %s)" % vgate, unit="A")

        self.reset()
        self.beeper.beep(0.3, 2400)

        self.busy = False
        return rt

    def playChord(self, notes=("C6", "E6", "G6"), durations=0.3):
        """Plays a chord on the Keithley.

        :param list notes: List of notes in scientific pitch notation, for
            instance ``['F4', 'Ab4', 'C4']`` for a f-minor chord in the 4th
            octave. Defaults to c-major in the 6th octave.
        :param durations: List of durations for each note in sec. If a single
            float is given, all notes will have the same duration. Defaults to
            0.3 sec.
        :type durations: float or list
        """

        freqs = [self._pitch_to_freq(p) for p in notes]
        if not isinstance(durations, list):
            durations = [durations] * len(freqs)

        for f, d in zip(freqs, durations):
            self.beeper.beep(d, f)

    @staticmethod
    def _pitch_to_freq(pitch):

        A4 = 440
        C4 = A4 * 2.0 ** (-9 / 12)

        names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        names_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

        octave = list(filter(lambda x: x in "0123456789", pitch))
        octave = "".join(octave)

        pitch = pitch.strip(octave)

        if octave == "":
            octave = 4
        else:
            octave = int(octave)

        try:
            steps = names_sharp.index(pitch)
        except ValueError:
            steps = names_flat.index(pitch)

        steps += 12 * (octave - 4)

        freq = C4 * 2.0 ** (steps / 12)

        return freq


class Keithley2600Factory(object):

    _instances = {}

    def __new__(cls, *args, **kwargs):
        """
        Create new instance for a new visa_address, otherwise return existing
        instance.
        """
        if args[0] in cls._instances:
            logger.debug("Returning existing instance with address '%s'." % args[0])

            return cls._instances[args[0]]
        else:
            logger.debug("Creating new instance with address '%s'." % args[0])
            instance = Keithley2600(*args, **kwargs)
            cls._instances[args[0]] = instance

            return instance
