import os
from importlib.resources import contents

from compass.ocean.tests.global_ocean.metadata import \
    add_mesh_and_init_metadata
from compass.model import run_model
from compass.testcase import TestCase
from compass.step import Step


class ForwardStep(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of global ocean test
    cases.

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run

    time_integrator : {'split_explicit', 'RK4'}
        The time integrator to use for the forward run

    resources_from_config : bool
        Whether to get ``ntasks``, ``min_tasks`` and ``openmp_threads`` from
        config options
    """
    def __init__(self, test_case, mesh, init, time_integrator, name='forward',
                 subdir=None, ntasks=None, min_tasks=None,
                 openmp_threads=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        ntasks : int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads the step will use
        """
        self.mesh = mesh
        self.init = init
        self.time_integrator = time_integrator
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        if (ntasks is None) != (openmp_threads is None):
            raise ValueError('You must specify both ntasks and openmp_threads '
                             'or neither.')
        self.resources_from_config = ntasks is None

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file(
            'compass.ocean.tests.global_ocean', 'namelist.forward')
        self.add_streams_file(
            'compass.ocean.tests.global_ocean', 'streams.forward')

        if mesh.with_ice_shelf_cavities:
            self.add_namelist_file(
                'compass.ocean.tests.global_ocean', 'namelist.wisc')

        if init.with_bgc:
            self.add_namelist_file(
                'compass.ocean.tests.global_ocean', 'namelist.bgc')
            self.add_streams_file(
                'compass.ocean.tests.global_ocean', 'streams.bgc')

        mesh_package = mesh.package
        mesh_package_contents = list(contents(mesh_package))
        mesh_namelists = ['namelist.forward',
                          f'namelist.{time_integrator.lower()}']
        for mesh_namelist in mesh_namelists:
            if mesh_namelist in mesh_package_contents:
                self.add_namelist_file(mesh_package, mesh_namelist)

        mesh_streams = ['streams.forward',
                        f'streams.{time_integrator.lower()}']
        for mesh_stream in mesh_streams:
            if mesh_stream in mesh_package_contents:
                self.add_streams_file(mesh_package, mesh_stream)

        if mesh.with_ice_shelf_cavities:
            initial_state_target = \
                f'{init.path}/ssh_adjustment/adjusted_init.nc'
        else:
            initial_state_target = \
                f'{init.path}/initial_state/initial_state.nc'
        self.add_input_file(filename='init.nc',
                            work_dir_target=initial_state_target)
        self.add_input_file(
            filename='forcing_data.nc',
            work_dir_target=f'{init.path}/initial_state/'
                            f'init_mode_forcing_data.nc')
        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{init.path}/initial_state/graph.info')

        self.add_model_as_input()

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self._get_resources()

    def constrain_resources(self, available_cores):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_cores)

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)
        add_mesh_and_init_metadata(self.outputs, self.config,
                                   init_filename='init.nc')

    def _get_resources(self):
        # get the these properties from the config options
        if self.resources_from_config:
            config = self.config
            self.ntasks = config.getint(
                'global_ocean', 'forward_ntasks')
            self.min_tasks = config.getint(
                'global_ocean', 'forward_min_tasks')
            self.openmp_threads = config.getint(
                'global_ocean', 'forward_threads')

class ForwardTestCase(TestCase):
    """
    A parent class for test cases for forward runs with global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run

    time_integrator : {'split_explicit', 'RK4'}
        The time integrator to use for the forward run
    """

    def __init__(self, test_group, mesh, init, time_integrator, name):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run

        name : str
            the name of the test case
        """
        self.mesh = mesh
        self.init = init
        self.time_integrator = time_integrator
        subdir = get_forward_subdir(init.init_subdir, time_integrator, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        self.init.configure(config=self.config)


def get_forward_subdir(init_subdir, time_integrator, name):
    """
    Get the subdirectory for a test case that is based on a forward run with
    a time integrator
    """
    if time_integrator == 'split_explicit':
        # this is the default so we won't make a subdir for the time
        # integrator
        subdir = os.path.join(init_subdir, name)
    elif time_integrator == 'RK4':
        subdir = os.path.join(init_subdir, time_integrator, name)
    else:
        raise ValueError(f'Unexpected time integrator {time_integrator}')

    return subdir
