# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:19:05 2016

@author: Sam Schott (ss2151@cam.ac.uk)

(c) Sam Schott; This work is licensed under a Creative Commons
Attribution-NonCommercial-NoDerivs 2.0 UK: England & Wales License.

"""
import re
import time
import matplotlib.pyplot as plt
from qtpy import QtWidgets
import numpy as np


class SweepData(object):
    """
    Class to handle and store two terminal transfer and ouput voltage sweep
    data. The raw data is stored as numpy vectors:

        self.Vg - gate voltage
        self.Vd - drain voltage
        self.Ig - gate current
        self.Id - drain current

    The following metadata is accessible:

        self.sweepType -    string that describes the sweep type, can be
                            'transfer' or 'output'
        self.vStep     -    stepped voltages (dynamically updated)
        self.nStep     -    number of voltage steps (dynamically updated)

    """

    def __init__(self, sweepType='transfer', Vg=[], Vd=[], Ig=[], Id=[]):
        TYPELIST = ['transfer', 'output']
        if sweepType in TYPELIST:
            self.sweepType = sweepType
        else:
            raise Exception('"sweepType" must be "transfer" or "output".')

        self.Vg = Vg
        self.Vd = Vd
        self.Ig = Ig
        self.Id = Id

        self._updateVstep()

    def append(self, Vg, Vd, Ig, Id):
        """
        Appends new voltage sweep data to the numpy vectors.
        """
        self.Vg = np.append(self.Vg, Vg)
        self.Vd = np.append(self.Vd, Vd)
        self.Ig = np.append(self.Ig, Ig)
        self.Id = np.append(self.Id, Id)

        self._updateVstep()

    def _updateVstep(self):
        # use rounded values
        VgRound = np.round(self.Vg, 1)
        VdRound = np.round(self.Vd, 1)

        if self.sweepType == 'transfer':
            unq, idx, counts = np.unique(VdRound, return_index=True,
                                         return_counts=True)
            self.vStep = unq[counts > 2].tolist()
            if any(counts == 2):
                self.vStep.append('Vg')
        elif self.sweepType == 'output':
            _, idx = np.unique(VgRound, return_index=True)
            self.vStep = VgRound[np.sort(idx)].tolist()

        self.nStep = len(self.vStep)

    def get_data_matrix(self):
        """
        Returns the voltage sweep data as a matrix of the form:
        [V_sweep, Id (step1), Ig (step1), Id (setp2), Ig (setp2), ...]
        """
        VgMatrix = np.reshape(self.Vg, (-1, self.nStep), order='F')
        VdMatrix = np.reshape(self.Vd, (-1, self.nStep), order='F')
        IgMatrix = np.reshape(self.Ig, (-1, self.nStep), order='F')
        IdMatrix = np.reshape(self.Id, (-1, self.nStep), order='F')

        if self.sweepType == 'transfer':
            Vg = VgMatrix[:, 0]
            matrix = np.concatenate((Vg[:, np.newaxis], IdMatrix, IgMatrix),
                                    axis=1)
        elif self.sweepType == 'output':
            Vd = VdMatrix[:, 0]
            matrix = np.concatenate((Vd[:, np.newaxis], IdMatrix, IgMatrix),
                                    axis=1)
        return matrix

    def plot(self):
        """
        Plots the transfer or output curves in a PyQt window. Do not call this
        function from within a thread.
        """
        if self.sweepType == 'transfer':
            fig1 = plt.figure(1)
            plt.clf()
            fig2 = plt.figure(2)
            plt.clf()

            for i in range(0, self.nStep):
                nPoints = len(self.Vg)/self.nStep
                select = slice(i*nPoints, (i+1)*nPoints)
                plt.figure(1)
                plt.semilogy(self.Vg[select], abs(self.Id[select]), '-',
                             label='Drain current, Vd = %s' % self.vStep[i])
                plt.semilogy(self.Vg[select], abs(self.Ig[select]), '--',
                             label='Gate current, Vd = %s' % self.vStep[i])
                plt.legend(loc=3)

                plt.figure(2)
                plt.plot(self.Vg[select], np.sqrt(abs(self.Id[select])))

            plt.show()
            fig1.canvas.draw()
            fig2.canvas.draw()

        if self.sweepType == 'output':
            fig1 = plt.figure(1)
            plt.clf()

            for i in range(0, self.nStep):
                nPoints = len(self.Vg)/self.nStep
                select = slice(i*nPoints, (i+1)*nPoints)
                plt.figure(1)
                plt.plot(self.Vd[select], abs(self.Id[select]), '-',
                         label='Drain current, Vg = %s' % self.vStep[i])
                plt.plot(self.Vd[select], abs(self.Ig[select]), '--',
                         label='Gate current, Vg = %s' % self.vStep[i])
                plt.legend()

            fig1.canvas.draw()

    def save(self, filepath=None):
        """
        Saves the votage sweep data as a text file with headers. If no filepath
        is given, the user is promted to select a location and name through a
        user interface.
        """
        # create header and title for file
        header = []
        time_str = time.strftime('%H:%M, %d/%m/%Y')
        if self.sweepType == 'output':
            title = '# output curve, recorded at %s\n' % time_str
            header.append('Vd /V')
            for i in range(0, self.nStep):
                header.append('Isd (Vg=%dV) /A' % self.vStep[i])
            for i in xrange(0, self.nStep):
                header.append('Ig (Vg=%dV) /A' % self.vStep[i])
        elif self.sweepType == 'transfer':
            title = '# transfer curve, recorded at %s\n' % time_str
            header.append('Vg /V')
            for i in range(0, self.nStep):
                header.append('Isd (Vd=%sV) /A' % self.vStep[i])
            for i in range(0, self.nStep):
                header.append('Ig (Vd=%sV) /A' % self.vStep[i])

        data_matrix = self.get_data_matrix()
        self.header = '\t'.join(header)

        # save to file
        if filepath is None:
            text = 'Please select file for sweep data:'
            filepath = QtWidgets.QFileDialog.getSaveFileName(caption=text)
            filepath = filepath[0]

        if len(filepath) > 4:
            np.savetxt(filepath, data_matrix, fmt='%.9E', delimiter='\t',
                       newline='\n', header=self.header, comments=title)

        return filepath

    def load(self, filepath=None):
        """
        Loads the votage sweep data from a text file. If no filepath is given,
        the user is promted to select a file through a GUI.
        """

        if filepath is None:
            text = 'Please select file with mode IV curve data:'
            filepath = QtWidgets.QFileDialog.getOpenFileName(caption=text)
            filepath = filepath[0]

        if len(filepath) > 4:

            # get info string and header
            with open(filepath) as f:
                info_string = f.readline().strip()
                header = f.readline().strip()

            # read in data
            data_matrix = np.loadtxt(filepath, skiprows=2)

            # determine scweep type (transfer / output), proceed accordingly
            if info_string.find('transfer') > 0:
                self.sweepType = 'transfer'
                number_of_sweeps = (data_matrix.shape[1] - 1)/2
                voltages = self._find_numbers(header)
                Vg = data_matrix[:, 0]

                for i in range(0, number_of_sweeps):
                    Id = data_matrix[:, i + 1]
                    Ig = data_matrix[:, i + 1 + number_of_sweeps]
                    Vd = np.ones(len(data_matrix)) * voltages[i]
                    self.append(Vg, Vd, Ig, Id)

            elif info_string.find('output') > 0:
                self.sweepType = 'output'
                number_of_sweeps = (data_matrix.shape[1] - 1)/2
                voltages = self._find_numbers(header)
                Vd = data_matrix[:, 0]

                for i in range(0, number_of_sweeps):
                    Id = data_matrix[:, i + 1]
                    Ig = data_matrix[:, i + 1 + number_of_sweeps]
                    Vg = np.ones(len(data_matrix)) * voltages[i]
                    self.append(Vg, Vd, Ig, Id)

            self._updateVstep()

    def _find_numbers(self, string):
        """
        Finds all numbers in a string, for example in a header from a txt file.
        """
        fmt = '[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?'
        string_list = re.findall(fmt, string)
        float_list = [float(s) for s in string_list]

        return float_list
