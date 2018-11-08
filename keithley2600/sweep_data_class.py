# -*- coding: utf-8 -*-
#
# Copyright © keithley2600 Project Contributors
# Licensed under the terms of the MIT License
# (see keithley2600/__init__.py for details)

"""
Submodule defining classes to hold, plot, and save IV and FET sweep data.

"""

import re
import time
import matplotlib.pyplot as plt
import numpy as np


class IVSweepData(object):
    """
    Class to store, load, save, and plot data from a simple IV sweep. The data
    is stored in two numpy arrays of voltage and current.
    """

    TYPELIST = ['iv', ]

    def __init__(self, v=None, i=None):

        self.sweepType = 'iv'

        if v is None:
            self.v, self.i = np.array([]), np.array([])
        else:
            self.v, self.i = np.array(v), np.array(i)

        assert len(self.v) == len(self.i)

    def get_data_matrix(self):
        """ Returns matrix with columns (v, i)."""

        m = np.array([self.v, self.i])

        return m.transpose()

    def append(self, v, i):

        assert len(i) == len(v)

        self.v = np.append(self.v, v)
        self.i = np.append(self.v, i)

    def plot(self, log=False):
        """Plots the IV data on a linear (default) or semilog scale. """

        fig1 = plt.figure(1)
        plt.clf()
        plt.figure(1)

        if log:
            plt.semilogy(self.v, self.i, '-')
        else:
            plt.plot(self.v, self.i, '-')

        fig1.canvas.draw()

    def save(self, filepath):
        """
        Saves the votage sweep data to a text file. The file format is:
            * The header contains the sweep type (iv) and date saved.
            * Tab delimited columns contain voltage and current data.
            * The first column contains voltages of the sweep SMU.
        """
        # create header and title for file
        time_str = time.strftime('%H:%M, %d/%m/%Y')

        title = '# iv curve, recorded at %s\n' % time_str
        header = ['V /V', 'I /A']

        data_matrix = self.get_data_matrix()
        header = '\t'.join(header)

        # save to file
        np.savetxt(filepath, data_matrix, fmt='%.9E', delimiter='\t',
                   newline='\n', header=header, comments=title)

        return filepath

    def load(self, filepath):
        """
        Loads the votage sweep data from a text file. The expected format is:
            * The header contains the sweep type (iv) and date saved.
            * Tab delimited columns contain voltage and current data.
            * The first column contains voltages of the sweep SMU.
        """
        # reset to empty values
        self.v, self.i = np.array([]), np.array([])

        # read info string and header
        with open(filepath) as f:
            info_string = f.readline().strip()
            header = f.readline().strip()

        # read data as 2D numpy array
        m = np.loadtxt(filepath, skiprows=2)

        # check file format
        headers = header.split('\t')
        if len(headers) != 2 or m.shape[1] != 2:
            raise RuntimeError('2 columns expected but %s ' % len(headers) +
                               'columns found. Please check file format.')

        # determine sweep type (transfer / output), proceed accordingly
        types_in_head = [t for t in self.TYPELIST if (t in info_string)]

        if len(types_in_head) == 1:
            self.sweepType = types_in_head[0]
        else:
            raise RuntimeError('File type not recognized. Please check if ' +
                               'the file contains valid IV sweep data.')

        self.v = m[:, 0]
        self.i = m[:, 1]


class TransistorSweepData(object):
    """
    Class to handle, store and load transfer and ouput characteristic data of
    FETs. The data is stored in dictionaries of numpy arrays with entries
    named after the fixed voltage:

        self.vSweep - dictionary of voltages on sweep SMU
        self.iSource - dictionary of source SMU currrents for all sweeps
        self.iDrain - dictionary of drain SMU currrents for all sweeps
        self.iGate - dictionary of gate SMU currrents for all sweeps

    The following metadata is accessible:

        self.sweepType   - string that describes the sweep type, can be
                           'transfer' or 'output'
        self.step_list() - returns list of stepped voltages
        self.n_steps()   - returns number of fixed voltage steps

    """

    TYPELIST = ['transfer', 'output']

    def __init__(self, sweepType='transfer', vSweep=None, iSource=None,
                 iDrain=None, iGate=None):

        if sweepType in self.TYPELIST:
            self.sweepType = sweepType
        else:
            raise RuntimeError('"sweepType" must be "transfer", "output".')

        if vSweep is None:
            self.vSweep = {}
            self.iSource, self.iDrain, self.iGate = {}, {}, {}
        else:
            self.vSweep = vSweep
            self.iSource, self.iDrain, self.iGate = iSource, iDrain, iGate

        # perform checks on data
        assert self.iSource.keys() == self.vSweep.keys()
        assert self.iDrain.keys() == self.vSweep.keys()
        assert self.iGate.keys() == self.vSweep.keys()

    def step_list(self):
        return list(self.vSweep.keys())

    def n_steps(self):
        return len(self.step_list())

    def append(self, vFix, vSweep, iSource=np.array([]), iDrain=np.array([]),
               iGate=np.array([])):
        """
        Appends new voltage sweep data to the numpy vectors. Calculates missing
        currents if necessary.
        """
        if not iSource.size:
            iSource = np.array(iGate) + np.array(iDrain)

        if vFix in self.vSweep.keys():
            # Append to existing sweep data if voltage step already exists
            self.vSweep[vFix] = np.append(self.vSweep[vFix], vSweep)
            self.iSource[vFix] = np.append(self.iSource[vFix], iSource)
            self.iDrain[vFix] = np.append(self.iDrain[vFix], iDrain)
            self.iGate[vFix] = np.append(self.iGate[vFix], iGate)
        else:
            # Create new entries for new step voltages
            self.vSweep[vFix] = np.array(vSweep)
            self.iSource[vFix] = np.array(iSource)
            self.iDrain[vFix] = np.array(iDrain)
            self.iGate[vFix] = np.array(iGate)

    def get_data_matrix(self):
        """
        Returns the voltage sweep data as a matrix of the form:
        [V_sweep, Is (step1), Id (step1), Ig (step1), Is (setp2), ...]
        """

        vSweepMatrix = np.array(list(self.vSweep.values()))

        iSMatrix = np.array(list(self.iSource.values()))
        iDMatrix = np.array(list(self.iDrain.values()))
        iGMatrix = np.array(list(self.iGate.values()))

        for m in (iSMatrix, iDMatrix, iGMatrix):
            m = self._pad2shape(m, vSweepMatrix.shape)

        vSweep0 = vSweepMatrix[0, :]

        matrix = np.concatenate((vSweep0[np.newaxis, :], iSMatrix,
                                 iDMatrix, iGMatrix))

        return matrix.transpose()

    def _pad2shape(self, array, shape):
        """Pads numpy array with NaN until shape is reached."""

        padded = np.empty(shape) * np.NaN
        padded[:array.shape[0], :array.shape[1]] = array

        return padded

    def plot(self):
        """
        Plots the transfer or output curves. This method is not thread safe.
        """
        if self.sweepType == 'transfer':
            fig1, fig2 = plt.figure(1), plt.figure(2)
            fig1.clf()
            fig2.clf()

            for v in self.vSweep.keys():
                # log plot
                plt.figure(1)
                plt.semilogy(self.vSweep[v], abs(self.iSource[v]), '-',
                             label='Source current, Vd = %s' % v)
                plt.semilogy(self.vSweep[v], abs(self.iDrain[v]), '-',
                             label='Drain current, Vd = %s' % v)
                plt.semilogy(self.vSweep[v], abs(self.iGate[v]), '--',
                             label='Gate current, Vd = %s' % v)
                plt.legend(loc=3)

                # sqrt(I) plot
                plt.figure(2)
                plt.plot(self.vSweep[v], np.sqrt(abs(self.iDrain[v])))

            plt.show()
            fig1.canvas.draw()
            fig2.canvas.draw()

        if self.sweepType == 'output':
            fig1 = plt.figure(1)
            plt.clf()

            for v in self.vSweep.keys():
                # linear plot
                plt.figure(1)
                plt.plot(self.vSweep[v], abs(self.iSource[v]), '-',
                         label='Source current, Vg = %s' % v)
                plt.plot(self.vSweep[v], abs(self.iDrain[v]), '--',
                         label='Drain current, Vg = %s' % v)
                plt.plot(self.vSweep[v], abs(self.iGate[v]), '--',
                         label='Gate current, Vg = %s' % v)
                plt.legend()

            fig1.canvas.draw()

    def save(self, filepath):
        """
        Saves the votage sweep data to a text file. The file format is:
            * The header contains the sweep type (transfer / output) and date
              saved.
            * Tab delimited columns of transfer or output sweep currents.
            * Fixed voltage steps are provided in the column titles.
            * The first column contains voltages of the sweep SMU.
            * Currents are saved in the order [Is, ... , Id, ... , Ig, ... ].
        """
        # create header and title for file
        time_str = time.strftime('%H:%M, %d/%m/%Y')

        if self.sweepType is 'output':
            title = '# output curve, recorded at %s\n' % time_str
            header = ['Vd /V']
            vFixName = 'Vg'
        elif self.sweepType is 'transfer':
            title = '# transfer curve, recorded at %s\n' % time_str
            header = ['Vg /V']
            vFixName = 'Vd'

        for i in ('Is', 'Id', 'Ig'):
            for v in self.vSweep.keys():
                header += ['%s (%s=%sV) /A' % (i, vFixName, v)]

        data_matrix = self.get_data_matrix()
        header = '\t'.join(header)

        # save to file
        np.savetxt(filepath, data_matrix, fmt='%.9E', delimiter='\t',
                   newline='\n', header=header, comments=title)

        return filepath

    def load(self, filepath):
        """
        Loads the votage sweep data from a text file. The exepected file format
        is:
            * The header contains the sweep type (transfer / output).
            * Tab delimited columns of transfer or output sweep currents
            * Fixed voltage steps must be provided in the column titles.
            * The first column must contain the sweep voltage.
            * Last columns are expected to contain the gate currents.
        """
        # reset to empty values
        self.vSweep, self.iSource, self.iDrain, self.iGate = {}, {}, {}, {}

        # read info string and header
        with open(filepath) as f:
            info_string = f.readline().strip()
            header = f.readline().strip()

        # read data as 2D numpy array
        m = np.loadtxt(filepath, skiprows=2)

        # process information
        column_title_volts = self._find_numbers(header)
        v_fix_list = list(set(column_title_volts))  # get voltage steps

        # determine sweep type (transfer / output), proceed accordingly
        types_in_head = [t for t in self.TYPELIST if (t in info_string)]

        if len(types_in_head) == 1:
            self.sweepType = types_in_head[0]
        else:
            raise RuntimeError('File type not recognized. Please check if ' +
                               'the file contains valid sweep data.')

        for v in v_fix_list:
            # find indices of columns that belong to the same sweep
            idx = [i + 1 for (i, x) in enumerate(column_title_volts) if x == v]
            # check if Id + Ig or Is + Id + Ig currets are provided
            if len(idx) == 2:
                self.append(vFix=v, vSweep=m[:, 0], iSource=np.array([]),
                            iDrain=m[:, idx[0]], iGate=m[:, idx[1]])
            elif len(idx) == 3:
                self.append(vFix=v, vSweep=m[:, 0], iSource=m[:, idx[0]],
                            iDrain=m[:, idx[1]], iGate=m[:, idx[2]])

    def _find_numbers(self, string):
        """
        Finds all numbers in a string and returns them in a list.
        """

        fmt = r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?'
        string_list = re.findall(fmt, string)
        float_list = [float(s) for s in string_list]

        return float_list
