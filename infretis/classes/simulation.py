from infretis.classes.tasks import Task
from infretis.core.common import (print_to_screen,
                                  task_from_settings,
                                  write_restart_file)

import copy
import os
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())

class Simulation:
    """This class defines a generic simulation.

    Attributes
    ----------
    cycle : dict of integers
        This dictionary stores information about the number of cycles.
        The keywords are:

         *  `step`: The current cycle number.
         *  `startcycle`: The cycle number we started at.
         *  `endcycle`: Represents the cycle number where the simulation
            should end.
         * `stepno`: The number of cycles we have performed to arrive at
            cycle number given by `cycle['step']`. Note that `cycle['stepno']`
            might be different from `cycle['step']` since `cycle['startcycle']`
            might be != 0.

    exe_dir : string
        The path we are running the simulation from.
    restart_freq : integer
        The frequency for creating restart files.
    first_step : boolean
        True if the first step has not been executed yet.
    system : object like :py:class:`.System`
        This is the system the simulation will act on.
    simulation_output : list of dicts
        This list defines the output tasks associated
        with the simulation.
    simulation_type : string
        An identifier for the simulation.
    tasks : list of objects like :py:class:`.SimulationTask`
        This is the list of simulation tasks to execute.

    """

    simulation_type = 'generic'
    simulation_output = []

    def __init__(self, settings, controls):
        """Initialise the simulation object.

        Parameters
        ----------
        controls: dict of parameters to set up and control the simulations.
            It contains:

             *  `steps`: int, optional
                The number of simulation steps to perform.
             *  `startcycle`: int, optional
                The cycle we start the simulation on, useful for restarts.
             *  `endcycle`: int, optional
                The cycle we end the simulation to, useful in restarts.
             *  `rgen`: object like :py:class:`.RandomGenerator`
                The random generator that will be used for the
                paths that required random numbers.

        settings : dict
            Contains all the simulation settings.

        """
        steps = controls.get('steps', 0)
        startcycle = controls.get('startcycle', 0)
        end = controls.get('endcycle', steps)
        self.cycle = {'step': startcycle, 'endcycle': end,
                      'startcycle': startcycle, 'stepno': 0, 'steps': steps}

        self.tasks = []
        self.output_tasks = []
        self.first_step = True
        self.system = None
        self.restart_freq = None
        self.exe_dir = None
        self.settings = settings

    def extend_cycles(self, steps):
        """Extend a simulation with the given number of steps.

        Parameters
        ----------
        steps :  int
            The number of steps to extend the simulation with.

        Returns
        -------
        out : None
            Returns `None` but modifies `self.cycle`.

        """
        self.cycle['startcycle'] = self.cycle['stepno']
        self.cycle['endcycle'] = self.cycle['startcycle'] + steps

    def is_finished(self):
        """Determine if the simulation is finished.

        In this object, the simulation is done if the current step
        number is larger than the end cycle. Note that the number of
        steps performed is dependent on the value of
        `self.cycle['startcycle']`.

        Returns
        -------
        out : boolean
            True if the simulation is finished, False otherwise.

        """
        return self.cycle['step'] >= self.cycle['endcycle']

    def step(self):
        """Execute a simulation step.

        Here, the tasks in :py:attr:`.tasks` will be executed
        sequentially.

        Returns
        -------
        out : dict
            This dictionary contains the results of the defined tasks.

        Note
        ----
        This function will have 'side effects' and update/change
        the state of other attached variables such as the system or
        other variables that are not explicitly shown. This is intended
        and the behavior is defined by the tasks in
        :py:attr:`.tasks`.

        """
        if not self.first_step:
            self.cycle['step'] += 1
            self.cycle['stepno'] += 1
        results = self.execute_tasks()
        if self.first_step:
            self.first_step = False
        return results

    def execute_tasks(self):
        """Execute all the tasks in sequential order.

        Returns
        -------
        results : dict
            The results from the different tasks (if any).

        """
        results = {'cycle': self.cycle.copy()}
        for task in self.tasks:
            if not self.first_step or task.run_first():
                resi = task.execute(self.cycle)
                if task.result is not None:
                    results[task.result] = resi
        results['system'] = self.system
        return results

    def add_task(self, task, position=None):
        """Add a new simulation task.

        A task can still be added manually by simply appending to
        py:attr:`.tasks`. This function will, however, do some
        checks so that the task added can be executed.

        Parameters
        ----------
        task : dict
            A dict defining the task. A task is represented by an
            object of type :py:class:`.SimulationTask` with some
            additional settings on how to store the output
            and when to execute the task.
            The keywords in the dict defining the task are:

            * `func`: Callable, this is a function to execute in the
              task.
            * `args`: List, with arguments for the function.
            * `kwargs`: Dict, with the keyword arguments for the
              function.
            * `when`: Dict, which defines when the task should be
              executed.
            * `first`: Boolean, determines if the task should be
              executed on the initial step, i.e. before the
              full simulation starts.
            * `result`: String, used to label the result.

        position : int, optional
            Can be used to give the tasks a specific position in the
            task list.

        """
        try:
            new_task = SimulationTask(task['func'],
                                      args=task.get('args', None),
                                      kwargs=task.get('kwargs', None),
                                      when=task.get('when', None),
                                      result=task.get('result', None),
                                      first=task.get('first', False))
            if position is None:
                self.tasks.append(new_task)
            else:
                self.tasks.insert(position, new_task)
            return True
        except AssertionError:
            logger.warning('Could not add task: %s', task)
            return False

    def run(self):
        """Run a simulation.

        The intended usage is for simulations where all tasks have
        been defined in :py:attr:`self.tasks`.

        Note
        ----
        This function will just run the tasks via executing
        :py:meth:`.step` In general, this is probably too generic for
        the simulation you want, if you are creating a custom simulation.
        Please consider customizing the :py:meth:`.run` (or the
        :py:meth:`.step`) method of your simulation class.

        Yields
        ------
        out : dict
            This dictionary contains the results from the simulation.

        """
        while not self.is_finished():
            result = self.step()
            for task in self.output_tasks:
                task.output(result)
            self.write_restart()
            if self.soft_exit():
                yield result
                break
            yield result

    def __str__(self):
        """Just a small function to return some info about the simulation."""
        ntask = len(self.tasks)
        mtask = 'task' if ntask == 1 else 'tasks'
        msg = ['Generic simulation with {} {}.'.format(ntask, mtask)]
        for i, task in enumerate(self.tasks):
            msg += ['* Task no. {}'.format(i)]
            for j, line in enumerate(str(task).split('\n')):
                if j > 0:
                    msg += [line]
        return '\n'.join(msg)

    def set_up_output(self, settings, progress=False):
        """Set up output from the simulation.

        This includes the predefined output tasks, but also output
        related to the restart file(s).


        Parameters
        ----------
        settings : dict
            These are the simulation settings.
        progress : boolean
            For some simulations, the user may select to display a
            progress bar, we then need to disable the screen output.

        """
        logging.debug('Setting up output for simulation %s',
                      self.__class__.__name__)
        # Create the output tasks:
        self.create_output_tasks(settings, progress=progress)
        # Do set-up for restart output:
        self.restart_freq = settings['output'].get('restart-file', -1)
        if self.restart_freq < 1:
            self.restart_freq = None
            logger.warning('Writing of restart file(s) disabled!')
        logger.debug('Setting restart frequency for simulation %s',
                     self.restart_freq)
        self.exe_dir = settings['simulation'].get('exe_path', os.getcwd())

    def create_output_tasks(self, settings, progress=False):
        """Create output tasks for the simulation.

        This method will generate output tasks based on the tasks
        listed in :py:attr:`.simulation_output`.

        Parameters
        ----------
        settings : dict
            These are the simulation settings.
        progress : boolean
            For some simulations, the user may select to display a
            progress bar, we then need to disable the screen output.

        """
        logging.debug('Clearing output tasks & adding pre-defined ones')
        engine = getattr(self, 'engine', None)
        order_function = getattr(self, 'order_function', None)
        self.output_tasks = []
        directory = settings['simulation'].get('exe_path', None)
        for task_dict in self.simulation_output:
            if 'order' in task_dict['type'] and order_function is None:
                continue
            task = task_from_settings(task_dict, settings, directory, engine,
                                      progress)
            if task is not None:
                logger.debug('Created output task:\n%s', task)
                self.output_tasks.append(task)

    def soft_exit(self):
        """Force simulation to stop at the current step."""
        exit_file = 'EXIT'
        if self.exe_dir:
            exit_file = os.path.join(self.exe_dir, exit_file)
        if os.path.isfile(exit_file):
            logger.info('Exit file found - will do a soft exit.')
            print_to_screen('Exit file found - will do a soft exit.',
                            level='warning')
            # Write restart file...
            self.write_restart(now=True)
            # Close output files...
            for task in self.output_tasks:
                if task.target == 'file':
                    task.writer.close()
            return True
        return False

    def write_restart(self, now=False):
        """Create a restart file.

        Parameters
        ----------
        now : boolean, optional
            If True, the output file will be written irrespective of the
            step number.

        """
        if now or (self.restart_freq is not None and
                   self.cycle['step'] % self.restart_freq == 0):
            out = 'pyretis.restart'
            if self.exe_dir:
                out = os.path.join(self.exe_dir, out)
            write_restart_file(out, self)

    def restart_info(self):
        """Return information which can be used to restart the simulation.

        Returns
        -------
        info : dict,
            Contains all the updated simulation settings and counters.

        """
        info = {}
        info['settings'] = self.settings

        for key in {'simulation', 'system', 'particles'}:
            if key not in info:
                info[key] = {}

        info['simulation']['restart'] = 'pyretis.restart'
        info['simulation']['cycle'] = copy.deepcopy(self.cycle)
        info['simulation']['type'] = self.simulation_type

        if hasattr(self, 'system') and self.system is not None:
            info['system'].update(self.system.restart_info())

        if hasattr(self.system, 'particles') and \
                self.system.particles is not None:
            info['particles'].update(self.system.particles.restart_info())

        return info

    def load_restart_info(self, info):
        """Load restart information.

        Note, we do not change the ``end`` property here as we probably
        are extending a simulation.

        Parameters
        ----------
        info : dict
            The dictionary with the restart information.

        """
        for key, val in info['simulation']['cycle'].items():
            if key in {'steps', 'endcycle'}:
                self.cycle['startcycle'] = copy.deepcopy(val)
            else:
                self.cycle[key] = copy.deepcopy(val)

        if info.get('system') is not None and\
                hasattr(self, 'system') and self.system is not None:
            self.system.load_restart_info(info['system'])


class SimulationTask(Task):
    """Representation of simulation tasks.

    This class defines a task object. A task is executed at specific
    points, at regular intervals etc. in a simulation. A task will
    typically provide a result, but it does not need to. It can simply
    just alter the state of the passed argument(s).

    Attributes
    ----------
    function : function
        The function to execute.
    when : dict
        Determines when the task should be executed.
    args : list
        List of arguments to the function.
    kwargs : dict
        The keyword arguments to the function.
    first : boolean
        True if this task should be executed before the first
        step of the simulation.
    result : string
        This is a label for the result created by the task.

    """

    def __init__(self, function, args=None, kwargs=None, when=None,
                 result=None, first=False):
        """Initialise the task.

        Parameters
        ----------
        function : callable
            The function to execute.
        args : list, optional
            List of arguments to the function.
        kwargs : dict, optional
            The keyword arguments to the function.
        when : dict, optional
            Determines if the task should be executed.
        result : string, optional
            This is a label for the result created by the task.
        first : boolean, optional
            True if this task should be executed before the first
            step of the simulation.

        """
        if not callable(function):
            msg = 'The given function for the task is not callable!'
            raise AssertionError(msg)
        ok_to_add = _check_args(function, given_args=args, given_kwargs=kwargs)
        if not ok_to_add:
            msg = 'Wrong arguments or keyword arguments!'
            raise AssertionError(msg)
        super().__init__(when)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._result = result
        self.first = first

    def execute(self, step):
        """Execute the task.

        Parameters
        ----------
        step : dict of ints
            The keys are:

            * 'step': the current cycle number.
            * 'startcycle': the cycle number at the start.
            * 'stepno': the number of cycles we have performed so far.

        Returns
        -------
        out : unknown type
            The result of running `self.function`.

        """
        args = self.args
        kwargs = self.kwargs
        if self.execute_now(step):
            if args is None:
                if kwargs is None:
                    return self.function()
                return self.function(**kwargs)
            if kwargs is None:
                return self.function(*args)
            return self.function(*args, **kwargs)
        return None

    @property
    def result(self):
        """Return the result label."""
        return self._result

    def run_first(self):
        """Return True if task should be executed before first step."""
        return self.first

    def task_dict(self):
        """Return a dict representing the task."""
        return {'func': self.function, 'args': self.args,
                'kwargs': self.kwargs, 'when': self.when,
                'result': self.result, 'first': self.first,
                'func-name': self.function.__name__}

    def __call__(self, step):
        """Execute the task.

        Parameters
        ----------
        step : dict of ints
            The keys are:

            * 'step': the current cycle number.
            * 'startcycle': the cycle number at the start.
            * 'stepno': the number of cycles we have performed so far.

        Returns
        -------
        out : unknown type
            The result of `self.execute(step)`.

        """
        return self.execute(step)

    def __str__(self):
        """Output info about the task."""
        msg = ['Task:']
        msg += [f' -> Function name: {self.function.__name__}']
        msg += [f' -> Function args: {self.args}']
        msg += [f' -> Function kwargs: {self.kwargs}']
        msg += [f' -> Execute when: {self.when}']
        msg += [f' -> Execute at first: {self.first}']
        msg += [f' -> Result: {self._result}']
        return '\n'.join(msg)
