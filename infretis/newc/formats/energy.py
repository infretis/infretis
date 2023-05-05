# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""Module for formatting energy data from PyRETIS.

Important classes defined here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EnergyFormatter (:py:class:`.EnergyFormatter`)
    A class for formatting energy data from PyRETIS.

EnergyPathFormatter (:py:class:`.EnergyPathFormatter`)
    A class for formatting energy data for paths.

EnergyFile (:py:class:`.EnergyFile`)
    A class for handling PyRETIS energy files.

EnergyPathFile (:py:class:`.EnergyPathFile`)
    A class for handling PyRETIS energy path files.

"""
import logging
import numpy as np
from infretis.newc.formats.formatter import OutputFormatter, FileIO, read_some_lines
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


__all__ = [
    'EnergyFormatter',
    'EnergyPathFormatter',
    'EnergyFile',
    'EnergyPathFile',
]


class EnergyFormatter(OutputFormatter):
    """A class for formatting energy data from PyRETIS.

    This class handles formatting of energy data.
    The data is formatted in 5 columns:

    1) Time, i.e. the step number.

    2) Potential energy.

    3) Kinetic energy.

    4) Total energy, should equal the sum of the two previous columns.

    5) Temperature.

    """

    # Format for the energy files:
    ENERGY_FMT = ['{:>10d}'] + 5*['{:>14.6f}']
    ENERGY_TERMS = ('vpot', 'ekin', 'etot', 'temp')
    HEADER = {'labels': ['Time', 'Potential', 'Kinetic', 'Total',
                         'Temperature'],
              'width': [10, 14]}

    def __init__(self, name='EnergyFormatter'):
        """Initialise the formatter for energy."""
        super().__init__(name, header=self.HEADER)

    def apply_format(self, step, energy):
        """Apply the energy format.

        Parameters
        ----------
        step : int
            The current simulation step.
        energy : dict
            A dict with energy terms to format.

        Returns
        -------
        out : string
            A string with the formatted energy data.

        """
        towrite = [self.ENERGY_FMT[0].format(step)]
        for i, key in enumerate(self.ENERGY_TERMS):
            value = energy.get(key, None)
            if value is None:
                towrite.append(self.ENERGY_FMT[i + 1].format(float('nan')))
            else:
                towrite.append(self.ENERGY_FMT[i + 1].format(float(value)))
        return ' '.join(towrite)

    def format(self, step, data):
        """Yield formatted energy data. See :py:meth:.`apply_format`."""
        yield self.apply_format(step, data)

    def load(self, filename):
        """Load entire energy blocks into memory.

        Parameters
        ----------
        filename : string
            The path/file name of the file we want to open.

        Yields
        ------
        data_dict : dict
            This is the energy data read from the file, stored in
            a dict. This is for convenience so that each energy term
            can be accessed by `data_dict['data'][key]`.

        """
        for blocks in read_some_lines(filename, line_parser=self.parse):
            data = np.array(blocks['data'])
            col = tuple(data.shape)
            col_max = min(col[1], len(self.ENERGY_TERMS) + 1)
            data_dict = {'comment': blocks['comment'],
                         'data': {'time': data[:, 0]}}
            for i in range(col_max-1):
                data_dict['data'][self.ENERGY_TERMS[i]] = data[:, i+1]
            yield data_dict


class EnergyPathFormatter(EnergyFormatter):
    """A class for formatting energy data for paths."""

    ENERGY_TERMS = ('vpot', 'ekin')
    HEADER = {'labels': ['Time', 'Potential', 'Kinetic'],
              'width': [10, 14]}

    def __init__(self):
        """Initialise."""
        super().__init__(name='EnergyPathFormatter')
        self.print_header = False

    def format(self, step, data):
        """Format the order parameter data from a path.

        Parameters
        ----------
        step : int
            The cycle number we are creating output for.
        data : tuple
            Here we assume that ``data[0]`` contains an object
            like :py:class:`.PathBase` and that ``data[1]`` is a
            string with the status for the path.

        Yields
        ------
        out : string
            The strings to be written.

        """
        path, status = data[0], data[1]
        if not path:  # when nullmoves = False
            return
        move = path.generated
        yield '# Cycle: {}, status: {}, move: {}'.format(step, status, move)
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            energy = {}
            for key in self.ENERGY_TERMS:
                energy[key] = getattr(phasepoint.particles, key, None)
            yield self.apply_format(i, energy)


class EnergyFile(FileIO):
    """A class for handling PyRETIS energy files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the file object and attach the energy formatter."""
        super().__init__(filename, file_mode, EnergyFormatter(), backup=backup)


class EnergyPathFile(FileIO):
    """A class for handling PyRETIS energy path files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the file object and attach the energy formatter."""
        super().__init__(filename, file_mode, EnergyPathFormatter(),
                         backup=backup)
