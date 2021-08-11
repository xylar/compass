#!/usr/bin/env python
import os
import pickle
import configparser

import mpas_tools.io

from compass.parallel import get_available_cores_and_nodes


def main():

    with open('test_case.pickle', 'rb') as handle:
        test_case = pickle.load(handle)

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(test_case.config_filename)
    steps_to_run = config.get('test_case',
                              'steps_to_run').replace(',', ' ').split()

    mpas_tools.io.default_format = config.get('io', 'format')
    mpas_tools.io.default_engine = config.get('io', 'engine')

    base_dir = os.getcwd()

    for step_name in steps_to_run:
        step = test_case.steps[step_name]
        step.config = config
        step_dir = os.path.join(base_dir, step.subdir)

        cores = _get_step_cores(step, config)
        step.cores = cores

        missing_files = list()
        for input_file in step.inputs:
            # this input doesn't seem to be a data future from another app,
            # so let's make sure it exists
            if not os.path.exists(input_file):
                missing_files.append(input_file)

        if len(missing_files) > 0:
            raise OSError(
                'input file(s) missing in step {} of {}/{}/{}: {}'.format(
                    step.name, step.mpas_core.name, step.test_group.name,
                    step.test_case.subdir, missing_files))

        print(f'Running: {step_name}')
        run_step(step_dir, step)


# pulled out of Testcase.run() and its helper method Testcase._run_step()
def run_step(step_dir, step):
    """
    Run a step in its work directory

    Parameters
    ----------
    step_dir : str
        The work directory for the step

    step : compass.step.Step
        The step object
    """
    import os
    from mpas_tools.logging import LoggingContext

    original_dir = os.getcwd()
    os.chdir(step_dir)

    # start logging to stdout/stderr
    test_name = step.path.replace('/', '_')
    with LoggingContext(name=test_name) as logger:
        step.logger = logger
        step.run()

    os.chdir(original_dir)


def _get_step_cores(step, config):
    """ set step.cores based on config options and available cores """

    cores = step.cores
    min_cores = step.min_cores

    # if this is a forward step, update the cores and min_cores from the
    # config file in case a user has changed them
    if 'forward' in step.name:
        resolution = step.resolution

        cores = config.getint('cosine_bell',
                              'QU{}_cores'.format(resolution))
        min_cores = config.getint('cosine_bell',
                                  'QU{}_min_cores'.format(resolution))

    available_cores, _ = get_available_cores_and_nodes(config)
    cores = min(cores, available_cores)
    if min_cores is not None:
        if cores < min_cores:
            raise ValueError(
                'Available cores for {} is below the minimum of {}'
                ''.format(cores, min_cores))

    return cores


if __name__ == '__main__':
    main()
