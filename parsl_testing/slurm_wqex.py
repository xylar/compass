#!/usr/bin/env python

import parsl
from parsl.config import Config
from parsl.app.app import python_app
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.executors import WorkQueueExecutor
import time


# App that generates a random number after a delay
@python_app
def generate(limit, delay,
             parsl_resource_specification={'cores': 1,
                                           'memory': 100,
                                           'disk': 10}):
    from random import randint
    import time
    time.sleep(delay)
    return randint(1, limit)


if __name__ == '__main__':
    compass_branch = '/home/ac.xylar/mpas-work/compass/parsl_res_test/'
    activation_script = 'load_dev_compass_1.0.0_chrysalis_intel_impi.sh'
    # Command to be run before starting a worker
    worker_init = f'source {compass_branch}/{activation_script}'

    config = Config(
        executors=[
            WorkQueueExecutor(
                label='Chrysalis_WQEX',
                autolabel=True,
                autocategory=True,
                provider=SlurmProvider(
                    partition='compute',  # Partition / QOS
                    nodes_per_block=1,
                    init_blocks=1,
                    worker_init=worker_init,
                    launcher=SrunLauncher(),
                    walltime='02:00:00',
                ),
            )
        ]
    )

    # load the Parsl config
    dfk = parsl.load(config)

    start = time.time()

    # Generate 16 random numbers between 1 and 10
    rand_nums = []
    for i in range(16):
        rand_nums.append(generate(10, i))

    # Wait for all apps to finish and collect the results
    outputs = [i.result() for i in rand_nums]

    # Print results
    print(outputs)

    print(f'Task finished in {time.time() - start} seconds')

    dfk.cleanup()
    parsl.clear()

