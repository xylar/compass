#!/usr/bin/env python

import sys
import parsl
from parsl.config import Config
from parsl.app.app import python_app
from parsl.providers import LocalProvider
from parsl.executors import WorkQueueExecutor, HighThroughputExecutor
from parsl.executors.threads import ThreadPoolExecutor

import time
import multiprocessing


@python_app
def sleeper(dur=1, **kwargs):
    """
    An app that sleeps for certain duration
    """
    import time
    time.sleep(dur)
    return dur


if __name__ == '__main__':

    # type of executor, 'WorkQueue' or 'ThreadPool'
    if len(sys.argv) > 1:
        executor_type = sys.argv[1]
    else:
        executor_type = 'WorkQueue'

    provider = LocalProvider()

    if executor_type == 'WorkQueue':
        print('WQEX: local_provider')
        config = Config(
            executors=[
                WorkQueueExecutor(
                    provider=provider
                )
            ]
        )

        kwargs = dict(parsl_resource_specification={
            'cores': 1, 'memory': 100, 'disk': 100})

    elif executor_type == 'ThreadPool':
        print(f'TPEX - cpu cores: {multiprocessing.cpu_count()}')
        config = Config(
            executors=[
                ThreadPoolExecutor(
                    max_threads=multiprocessing.cpu_count(),
                    label='local_threads'
                )
            ]
        )

        kwargs = dict()
    elif executor_type == 'HighThroughput':
        print(f'HTEX - cpu cores: {multiprocessing.cpu_count()}')
        config = Config(
            executors=[
                HighThroughputExecutor(
                    max_workers=multiprocessing.cpu_count(),
                    label='local_threads',
                    provider=provider
                )
            ]
        )

        kwargs = dict()
    else:
        raise ValueError(f'Unexpected executor_type {executor_type}')

    # Load the Parsl config
    dfk = parsl.load(config)

    start = time.time()

    tasks = [sleeper(**kwargs) for i in range(5)]
    outputs = [task.result() for task in tasks]

    # Print results
    print(outputs)

    print(f'Task finished in {time.time() - start} seconds')

    dfk.cleanup()
    parsl.clear()
