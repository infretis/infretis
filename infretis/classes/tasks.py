class Task:
    """Base representation of a "task".

    A task is just something that is supposed to be executed at
    a certain point. This class will just set up functionality
    that is common for output tasks and for simulation tasks.

    Attributes
    ----------
    when : dict
        Determines when the task should be executed.

    """

    _ALLOWED_WHEN = {'every', 'at'}

    def __init__(self, when):
        """Initialise the task.

        Parameters
        ----------
        when : dict, optional
            Determines if the task should be executed.

        """
        self._when = None
        self.when = when

    @property
    def when(self):
        """Return the "when" property."""
        return self._when

    @when.setter
    def when(self, when):
        """Update self.when to new value(s).

        It will only update `self.when` for the keys given in the
        input `when`.

        Parameters
        ----------
        when : dict
            This dict contains the settings to update.

        Returns
        -------
        out : None
            Returns `None` but modifies `self.when`.

        """
        if when is None:
            self._when = when
        else:
            if self._when is None:
                self._when = {}
            for key, val in when.items():
                if key in self._ALLOWED_WHEN:
                    self._when[key] = val
                else:
                    logger.warning(
                        'Ignoring unknown "when" setting: "%s"', key
                    )

    def execute_now(self, step):
        """Determine if a task should be executed.

        Parameters
        ----------
        step : dict of ints
            Keys are 'step' (current cycle number), 'start' cycle number at
            start 'stepno' the number of cycles we have performed so far.

        Returns
        -------
        out : boolean
            True of the task should be executed.

        """
        if self.when is None:
            return True
        exe = False
        if 'every' in self.when:
            exe = step['stepno'] % self.when['every'] == 0
        if not exe and 'at' in self.when:
            try:
                exe = step['step'] in self.when['at']
            except TypeError:
                exe = step['step'] == self.when['at']
        return exe

    def task_dict(self):
        """Return basic info about the task."""
        return {'when': self.when}

class OutputTask(Task):
    """A base class for simulation output.

    This class will handle an output task for a simulation. The
    output task consists of one object which is responsible for
    formatting the output data and one object which is responsible
    for writing that data, for instance to the screen or to a file.

    Attributes
    ----------
    target : string
        This string identifies what kind of output we are dealing with.
        This will typically be either "screen" or "file".
    name : string
        This string identifies the task, it can, for instance, be used
        to reference the dictionary used to create the writer.
    result : tuple of strings
        This string defines the result we are going to output.
    writer : object like :py:class:`.OutputBase`
        This object will handle the actual outputting
        of the result.
    when : dict
        Determines if the task should be executed.

    """

    def __init__(self, name, result, writer, when):
        """Initialise the generic output task.

        Parameters
        ----------
        name : string
            This string identifies the task, it can, for instance, be used
            to reference the dictionary used to create the writer.
        result : list of strings
            These strings define the results we are going to output.
        writer : object like :py:class:`.IOBase`
            This object will handle formatting of the actual result and
            output to screen or to a file.
        when : dict
            Determines when the output should be written. Example:
            `{'every': 10}` will be executed at every 10th step.

        """
        super().__init__(when)
        self.target = writer.target
        self.name = name
        self.result = result
        self.writer = writer

    def output(self, simulation_result):
        """Output given results from simulation steps.

        This will output the task using the result found in the
        `simulation_result` which should be the dictionary returned
        from a simulation object (e.g. object like
        :py:class:`.Simulation`) after a step.

        Parameters
        ----------
        simulation_result : dict
            This is the result from a simulation step.

        Returns
        -------
        out : boolean
            True if the writer wrote something, False otherwise.

        """
        step = simulation_result['cycle']
        if not self.execute_now(step):
            return False
        result = []
        for res in self.result:
            if res not in simulation_result:
                return False  # Requested result was not ready at this step.
            result.append(simulation_result[res])
        if len(result) == 1:
            return self.writer.output(step['step'], result[0])
        return self.writer.output(step['step'], result)

    def __str__(self):
        """Print information about the output task."""
        msg = ['Output task {}'.format(self.name)]
        msg += ['- Output frequency: {}'.format(self.when)]
        msg += ['- Acting on result(s): {}'.format(self.result)]
        msg += ['- Target is: {}'.format(self.target)]
        msg += ['- Writer is: {}'.format(self.writer)]
        return '\n'.join(msg)

    def task_dict(self):
        """Return a dict with info about the task."""
        return {'name': self.name,
                'when': self.when,
                'result': self.result,
                'target': self.target,
                'writer': self.writer.__class__,
                'formatter': self.writer.formatter_info()}
