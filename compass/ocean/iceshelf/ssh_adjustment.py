import numpy
import xarray
import os

from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.testcase import TestCase
from compass.step import Step
from compass.model import ModelStep


class SshAdjustment(TestCase):
    """
    A test case for adjusting the sea-surface height or land-ice pressure to
    diminish the initial transient in simulations with ice-shelf cavities.

    Attributes
    ----------
    variable : {'ssh', 'landIcePressure'}
        The variable to adjust

    init_target_filename : str
        The relative path to the initial condition (either before SSH
        adjustment for iteration 0 or from the previous iteration)

    graph_target_filename : str
        The relative path to the graph file (typically created with the
        initial condition)

    output_filename : str
        The final adjusted output file name produced by the test case
    """
    def __init__(self, test_group, variable, init_target_filename,
                 graph_target_filename, name='ssh_adjustment', subdir=None):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.TestGroup
            the test group that this test case belongs to

        variable : {'ssh', 'landIcePressure'}
            The variable to adjust

        init_target_filename : str
            The relative path to the initial condition (either before SSH
            adjustment for iteration 0 or from the previous iteration)

        graph_target_filename : str
            The relative path to the graph file (typically created with the
            initial condition)

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the test case.  The default is ``name``
        """
        super().__init__(test_group=test_group, name=name, subdir=subdir)
        if variable not in ['ssh', 'landIcePressure']:
            raise ValueError(f'Unknown variable to modify: {variable}')
        self.variable = variable
        self.init_target_filename = init_target_filename
        self.graph_target_filename = graph_target_filename
        self.output_filename = None

    def configure(self):
        """
        Add the steps based on the ``iteration_count`` config option
        """
        config = self.config
        section = config['ssh_adjustment']
        iteration_count = section.getint('iterations')
        forward_ntasks = section.getint('forward_ntasks')
        forward_min_tasks = section.getint('forward_min_tasks')
        forward_threads = section.getint('forward_threads')
        forward_max_memory = section.getint('forward_max_memory')
        update_max_memory = section.getint('update_max_memory')

        variable = self.variable
        init_target_filename = self.init_target_filename
        graph_target_filename = self.graph_target_filename

        for iteration in range(iteration_count):
            step = SshAdjustmentForward(
                test_case=self, init_target_filename=init_target_filename,
                graph_target_filename=graph_target_filename,
                iteration=iteration, ntasks=forward_ntasks,
                min_tasks=forward_min_tasks, openmp_threads=forward_threads,
                max_memory=forward_max_memory)
            self.add_step(step)

            final = iteration == iteration_count-1
            step = SshAdjustmentUpdateInitialCondition(
                variable=variable, test_case=self, iteration=iteration,
                max_memory=update_max_memory, final=final)
            self.add_step(step)

            # the next target
            init_target_filename = f'../update{iteration:02d}/adjusted_init.nc'

        self.output_filename = \
            f'update{iteration_count - 1:02d}/adjusted_init.nc'


class SshAdjustmentForward(ModelStep):
    """
    A short forward run used in iterative adjustment of the

    Attributes
    ----------
    iteration : int
        The iteration number for this step
    """
    def __init__(self, test_case, init_target_filename, graph_target_filename,
                 iteration, ntasks, min_tasks, openmp_threads, max_memory):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        init_target_filename : str
            The relative path to the initial condition (either before SSH
            adjustment for iteration 0 or from the previous iteration)

        graph_target_filename : str
            The relative path to the graph file (typically created with the
            initial condition)

        iteration : int
            The iteration number for this step

        ntasks : int
            the target number of tasks the step would ideally use.  If too
            few cores are available on the system to accommodate the number of
            tasks and the number of cores per task, the step will run on
            fewer tasks as long as as this is not below ``min_tasks``

        min_tasks : int
            the number of tasks the step requires.  If the system has too
            few cores to accommodate the number of tasks and cores per task,
            the step will fail

        openmp_threads : int
            the number of OpenMP threads to use

        max_memory : int
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism

        """
        name = f'forward{iteration:02d}'
        if min_tasks is None:
            min_tasks = ntasks
        partition_graph = iteration == 0
        super().__init__(test_case=test_case, name=name, ntasks=ntasks,
                         min_tasks=min_tasks, openmp_threads=openmp_threads,
                         max_memory=max_memory, update_pio=True,
                         partition_graph=partition_graph)

        self.add_input_file(filename=f'init.nc',
                            target=init_target_filename)

        self.add_input_file(filename='graph.info',
                            target=graph_target_filename)

        self.iteration = iteration
        self.add_output_file(filename='output_ssh.nc')

    def setup(self):
        """
        Set up some namelist options that we want to take priority
        """
        # we want a 1-hour run and no freshwater fluxes under the ice shelf
        # from these namelist options
        self.add_namelist_file('compass.ocean.iceshelf', 'namelist.ssh_adjust')
        self.add_streams_file('compass.ocean.iceshelf', 'streams.ssh_adjust')

        super().setup()

    def runtime_setup(self):
        """
        Make a symlink to the graph partition if this is not the first
        iteration
        """

        super().runtime_setup()

        if self.iteration > 0:
            partition_filename = f'graph.info.part.{self.ntasks}'
            symlink(f'../forward00/{partition_filename}', partition_filename)


class SshAdjustmentUpdateInitialCondition(Step):
    """
    A step for adjusting the sea-surface height or land-ice pressure based
    on the output of the forward run

    Attributes
    ----------
    iteration : int
        The iteration number for this step

    variable : {'ssh', 'landIcePressure'}
        The variable to adjust

    final : bool
        Whether this is the final iteration, so this steps should also
        link an output file

    """
    def __init__(self, variable, test_case, iteration, max_memory, final):
        """
        Create the step

        Parameters
        ----------
        variable : {'ssh', 'landIcePressure'}
            The variable to adjust

        test_case : compass.TestCase
            The test case this step belongs to

        iteration : int
            The iteration number for this step

        max_memory : int
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism

        final : bool
            Whether this is the final iteration, so this steps should also
            link an output file
        """
        name = f'update{iteration:02d}'
        super().__init__(test_case=test_case, name=name, max_memory=max_memory)

        self.add_input_file(filename='init.nc',
                            target=f'../forward{iteration:02d}/init.nc')
        self.add_input_file(filename='output_ssh.nc',
                            target=f'../forward{iteration:02d}/output_ssh.nc')

        self.add_output_file(filename='adjusted_init.nc')
        if final:
            self.add_output_file(filename='../output/adjusted_init.nc')

        self.iteration = iteration
        self.variable = variable
        self.final = final

    def run(self):
        """
        Update the SSH or land-ice pressure based on the results of the
        forward run
        """
        logger = self.logger
        iteration = self.iteration

        logger.info("   * Updating SSH or land-ice pressure")

        try:
            os.makedirs('../output')
        except FileExistsError:
            pass

        with xarray.open_dataset('init.nc') as ds:

            # keep the data set with Time for output
            ds_out = ds

            ds = ds.isel(Time=0)

            on_a_sphere = ds.attrs['on_a_sphere'].lower() == 'yes'

            init_ssh = ds.ssh
            if 'minLevelCell' in ds:
                min_level_cell = ds.minLevelCell-1
            else:
                min_level_cell = xarray.zeros_like(ds.maxLevelCell)

            with xarray.open_dataset('output_ssh.nc') as ds_ssh:
                # get the last time entry
                ds_ssh = ds_ssh.isel(Time=ds_ssh.sizes['Time'] - 1)
                final_ssh = ds_ssh.ssh
                top_density = ds_ssh.density.isel(nVertLevels=min_level_cell)

            mask = numpy.logical_and(ds.maxLevelCell > 0,
                                     ds.modifyLandIcePressureMask == 1)

            delta_ssh = mask * (final_ssh - init_ssh)

            # then, modify the SSH or land-ice pressure
            if self.variable == 'ssh':
                ssh = final_ssh.expand_dims(dim='Time', axis=0)
                ds_out['ssh'] = ssh
                # also update the landIceDraft variable, which will be used to
                # compensate for the SSH due to land-ice pressure when
                # computing sea-surface tilt
                ds_out['landIceDraft'] = ssh
                # we also need to stretch layerThickness to be compatible with
                # the new SSH
                stretch = ((final_ssh + ds.bottomDepth) /
                           (init_ssh + ds.bottomDepth))
                ds_out['layerThickness'] = ds_out.layerThickness * stretch
                land_ice_pressure = ds.landIcePressure.values
            else:
                # Moving the SSH up or down by delta_ssh would change the
                # land-ice pressure by density(SSH)*g*delta_ssh. If delta_ssh
                # is positive (moving up), it means the land-ice pressure is
                # too small and if delta_ssh is negative (moving down), it
                # means land-ice pressure is too large, the sign of the second
                # term makes sense.
                gravity = constants['SHR_CONST_G']
                delta_land_ice_pressure = top_density * gravity * delta_ssh

                land_ice_pressure = numpy.maximum(
                    0.0, ds.landIcePressure + delta_land_ice_pressure)

                ds_out['landIcePressure'] = \
                    land_ice_pressure.expand_dims(dim='Time', axis=0)

                final_ssh = init_ssh

            write_netcdf(ds_out, 'adjusted_init.nc')

            # Write the largest change in SSH and its lon/lat to a file
            with open(f'../output/maxDeltaSSH_{iteration:03d}.log', 'w') as \
                    log_file:

                mask = land_ice_pressure > 0.
                cell_index = numpy.abs(delta_ssh.where(mask)).argmax().values

                ds_cell = ds.isel(nCells=cell_index)

                if on_a_sphere:
                    coords = 'lon/lat: {:f} {:f}'.format(
                        numpy.rad2deg(ds_cell.lonCell.values),
                        numpy.rad2deg(ds_cell.latCell.values))
                else:
                    coords = 'x/y: {:f} {:f}'.format(
                        1e-3 * ds_cell.xCell.values,
                        1e-3 * ds_cell.yCell.values)
                string = 'deltaSSHMax: {:g}, {}'.format(
                    delta_ssh.isel(nCells=cell_index).values, coords)
                logger.info('     {}'.format(string))
                log_file.write('{}\n'.format(string))
                string = 'ssh: {:g}, landIcePressure: {:g}'.format(
                    final_ssh.isel(nCells=cell_index).values,
                    land_ice_pressure.isel(nCells=cell_index).values)
                logger.info('     {}'.format(string))
                log_file.write('{}\n'.format(string))

        if self.final:
            # this is the final step
            symlink(f'../update{iteration:02d}/adjusted_init.nc',
                    '../output/adjusted_init.nc')

        logger.info("   - Complete\n")
