
import parsl
from parsl.config import Config
from parsl.app.app import python_app
from parsl.providers import LocalProvider
from parsl.providers import SlurmProvider
from parsl.executors import WorkQueueExecutor
from parsl.executors.threads import ThreadPoolExecutor

import os
import time
import glob
import json
import pandas as pd
import multiprocessing


# borrowed from PARSL tutorials to parse WQEX logfiles
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


# App that generates a random number after a delay
@python_app
def generate(task, path):
    #parsl_resource_specification={'cores': 1}):
    
    import numpy as np
    from mpas_tools.cime.constants import constants
    import mpas_tools.mesh.creation.build_mesh as mesh
    import os
    import shutil
    
    mesh_file = f"base_mesh_{task}.nc"
    mesh_file = os.path.join(path, "tmp", mesh_file)

    temp_path = f"tmp_{task}"
    temp_path = os.path.join(path, temp_path)

    this_path = os.getcwd()

    row = 180
    col = 360

    lon = np.linspace(-180., +180., col)
    lat = np.linspace(-90.0, +90.0, row)

    cellWidth = 480. * np.ones((row, col), dtype=np.float64)
    cellWidth = cellWidth / (task + 1)

    radius = constants["SHR_CONST_REARTH"] / 1.E+003

    shutil.rmtree(temp_path, ignore_errors=True)
    os.mkdir(temp_path)
    os.chdir(temp_path)  # workaround

    mesh.build_spherical_mesh(
        cellWidth, lon, lat, radius,
        out_filename=mesh_file, 
        plot_cellWidth=False, 
       #dir=os.path.join(path, "tmp")  # this doesn't seem to be working
    )
 
    os.chdir(this_path)  # workaround

    return task


if __name__ == '__main__':
    print("cpu cores:", multiprocessing.cpu_count())

    """
    config = Config(    
        executors=[
            WorkQueueExecutor(
               #port=50055,
                autolabel=True,
                autocategory=True,
                provider=LocalProvider()
            )
        ]
    )
    """

    config = Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=8,
                label='local_threads'
            )
        ]
    )

    HERE = os.path.abspath(os.path.dirname(__file__))

    # load the Parsl config
    dfk = parsl.load(config)

    start = time.time()

    # Generate 16 random numbers between 1 and 10
    task_list = []
    for i in range(2):
        task_list.append(generate(i, HERE))

    # Wait for all apps to finish and collect the results
    outputs = [task.result() for task in task_list]

    # Print results
    print(outputs)

    print(f'Task finished in {time.time() - start} seconds')

   #print(parse_logs())

    dfk.cleanup()
    parsl.clear()
