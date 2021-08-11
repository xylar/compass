#!/usr/bin/env python
import os
import pickle
import configparser
import glob
import json
import pandas as pd

import mpas_tools.io

import parsl
from parsl.data_provider.files import File
from parsl.app.app import python_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import WorkQueueExecutor

from compass.parallel import get_available_cores_and_nodes


def main():
    provider = LocalProvider()
    parsl_config = Config(
        executors=[
            WorkQueueExecutor(
                provider=provider,
                shared_fs=True,
                autolabel=False,
                autocategory=False
            )
        ]
    )

    # Load the Parsl config
    dfk = parsl.load(parsl_config)

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

    apps = dict()
    data_futures = dict()

    for step_name in steps_to_run:
        step = test_case.steps[step_name]
        step.config = config
        step_dir = os.path.join(base_dir, step.subdir)

        step.cores = _get_step_cores(step, config)
        # the current specs aren't big enough, so hard-code for now
        # memory = step.max_memory
        memory = 10000
        resources = {'cores': step.cores, 'memory': memory,
                     'disk': step.max_disk}

        # to do memory profiling, just set the number of cores and set
        # autolabel=True and autocategory=True above
        # resources = {'cores': step.cores}

        inputs = list()
        missing_files = list()
        for input_file in step.inputs:

            if input_file in data_futures:
                inputs.append(data_futures[input_file])
            else:
                # this input doesn't seem to be a data future from another app,
                # so let's make sure it exists
                if not os.path.exists(input_file):
                    missing_files.append(input_file)
                else:
                    inputs.append(File(input_file))

        if len(missing_files) > 0:
            raise OSError(
                'input file(s) missing in step {} of {}/{}/{}: {}'.format(
                    step.name, step.mpas_core.name, step.test_group.name,
                    step.test_case.subdir, missing_files))

        outputs = [File(output) for output in step.outputs]

        print(f'\n\nLaunching: {step_name}\n\n')
        app = run_step(step_dir, step, inputs=inputs, outputs=outputs,
                       parsl_resource_specification=resources)

        # add the output DataFuture objects to the dictionary of data futures
        # for later steps to use
        for index, output in enumerate(app.outputs):
            data_futures[step.outputs[index]] = output

        apps[step_name] = app

    for step_name in apps:
        # make sure all the apps finish
        apps[step_name].result()
        print(f'\n\nDone: {step_name}\n\n')

    dfk.cleanup()
    parsl.clear()

    # df = parse_logs()
    # print(df)


# pulled out of Testcase.run() and its helper method Testcase._run_step()
@python_app
def run_step(step_dir, step, inputs=[], outputs=[],
             parsl_resource_specification={'cores': 1,
                                           'memory': 1000,
                                           'disk': 1000}):
    """
    Run a step in its work directory

    Parameters
    ----------
    step_dir : str
        The work directory for the step

    step : compass.step.Step
        The step object

    inputs : list
        The list of input Files or DataFutures

    outputs : list
        The list of output Files

    parsl_resource_specification : dict
        The resources used by the WorkQueueExecutor
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


# borrowed from Parsl tutorials to parse WQEX logfiles
def parse_logs():
    """
    Parse the resource assignment of Work Queue from the runinfo logs
    """
    dirs = glob.glob("runinfo/*")
    log = "{}/WorkQueueExecutor/transaction_log".format(sorted(dirs)[-1])
    with open(log) as f:
        lines = f.readlines()

    resources = ['task_id', 'task_type', 'cores', 'memory', 'disk']
    df = pd.DataFrame(columns=resources)
    task_ids = {}
    for line in lines:
        if "WAITING" in line and \
           "WAITING_RETRIEVAL" not in line and 'taskid' not in line:
                line = line.split()
                task_id = line[3]
                task_category = line[5]
                task_ids[task_id] = task_category

        # timestamp master-pid TASK id (continue next line)
        # DONE SUCCESS exit-code exceeded measured
        if "RUNNING" in line and 'taskid' not in line:
            line = line.split()
            task_id = line[3]
            s = json.loads(line[-1])

            # append the new resources to the panda's data frame.
            # Resources are represented in a json array as
            # [value, "unit", such as [1000, "MB"],
            # so we only take the first element of the array:
            df.loc[len(df)] = [task_id, task_ids[task_id]] \
                            + list(float(s[r][0]) for r in resources[2:])
    return df


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
