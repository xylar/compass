import sys
import os


def get_default(module, description, steps, subdir=None):
    """
    Set up a default dictionary describing the testcase in the given module.
    The dictionary contains the full name of the python module for the testcase,
    the name of the testcase (the final part of the full module name), the
    subdirectory for the testcase (the same as the ``name`` if not supplied),
    the names of the ``configure()`` and ``run()`` functions within the module,
    and empty lists of ``inputs`` and ``outputs``, the ``core`` and
    ``configuration`` of the testcase (the parsed from the full module name),
    the description of the testcase provided, and the dictionary of steps.

    Parameters
    ----------
    module : str
        The full name of the python module for the testcase, usually supplied
        from ``__name__``

    description : str
        A description of the testcase

    steps : dict
        A dictionary of steps within the testcase, with the names of each step
        as keys and a dictionary of information on the step as values.  Each
        step's dictionary must contain, at a minimum, the information added by
        :py:func:`compass.step.get_default()`

    subdir : str, optional
        The subdirectory for the testcase, which defaults to the name of the
        testcase (parsed from the module name).  If a testcase supports
        various parameter values, such as various resolutions, it may be
        useful to supply a subdirectory so that the location of each variant of
        the testcase is placed in a unique working directory

    Returns
    -------
    testcase : dict
        A dictionary with the default information about the testcase, most of
        which can be modified as appropriate
    """
    name = module.split('.')[-1]
    core = module.split('.')[1]
    configuration = module.split('.')[3]
    if subdir is None:
        subdir = name
    path = os.path.join(core, configuration, subdir)
    for step in steps.values():
        step['testcase'] = name
    if hasattr(sys.modules[module], 'configure'):
        configure = 'configure'
    else:
        configure = None
    testcase = {'module': module,
                'name': name,
                'path': path,
                'core': core,
                'configuration': configuration,
                'subdir': subdir,
                'description': description,
                'steps': steps,
                'configure': configure,
                'run': 'run'}
    return testcase


def run_steps(testcase, config, steps):
    """
    Run the requested steps of a testcase

    Parameters
    ----------
    testcase : dict
        The dictionary describing the testcase with info from
        :py:func:`compass.testcase.get_default()` and any additional information
        added when collecting and setting up the testcase.

    config : configparser.ConfigParser
        Configuration options for this testcase

    steps : list
        A list of the names of the steps from ``testcase['steps']`` to run
    """
    cwd = os.getcwd()
    for step_name in steps:
        step = testcase['steps'][step_name]
        run = getattr(sys.modules[step['module']], step['run'])
        os.chdir(step['work_dir'])
        run(step, config)
    os.chdir(cwd)