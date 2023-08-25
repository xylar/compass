from compass.ocean.tests.global_ocean.dynamic_adjustment import (
    DynamicAdjustment,
)
from compass.ocean.tests.global_ocean.forward import ForwardStep


class RRS6to18DynamicAdjustment(DynamicAdjustment):
    """
    A test case performing dynamic adjustment (dissipating fast-moving waves)
    from an initial condition on the RRS6to18 MPAS-Ocean mesh

    Attributes
    ----------
    restart_filenames : list of str
        A list of restart files from each dynamic-adjustment step
    """

    def __init__(self, test_group, mesh, init, time_integrator):
        """
        Create the test case

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
        """
        if time_integrator != 'split_explicit':
            raise ValueError(f'{mesh.mesh_name} dynamic adjustment not '
                             f'defined for {time_integrator}')

        restart_times = ['0001-01-01_06:00:00', '0001-01-01_12:00:00',
                         '0001-01-02_00:00:00', '0001-01-03_00:00:00',
                         '0001-01-04_00:00:00', '0001-01-07_00:00:00',
                         '0001-01-13_00:00:00', '0001-01-31_00:00:00']

        restart_filenames = [
            f'restarts/rst.{restart_time.replace(":", ".")}.nc'
            for restart_time in restart_times]

        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         restart_filenames=restart_filenames)

        module = self.__module__

        shared_options = \
            {'config_AM_globalStats_enable': '.true.',
             'config_AM_globalStats_compute_on_startup': '.true.',
             'config_AM_globalStats_write_on_startup': '.true.',
             'config_use_activeTracers_surface_restoring': '.true.'}

        # first step
        step_name = 'damped_adjustment_1'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-00_06:00:00'",
            'config_dt': "'00:00:30'",
            'config_btr_dt': "'00:00:01'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '1.0e-3'}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-00_06:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_output_file(filename=f'../{restart_filenames[0]}')
        self.add_step(step)

        # second step
        step_name = 'damped_adjustment_2'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-00_06:00:00'",
            'config_dt': "'00:01:00'",
            'config_btr_dt': "'00:00:02'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '4.0e-4',
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[0]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-00_06:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[0]}')
        step.add_output_file(filename=f'../{restart_filenames[1]}')
        self.add_step(step)

        # third step
        step_name = 'damped_adjustment_3'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-00_12:00:00'",
            'config_dt': "'00:02:00'",
            'config_btr_dt': "'00:00:04'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '1.0e-4',
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[1]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-00_12:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[1]}')
        step.add_output_file(filename=f'../{restart_filenames[2]}')
        self.add_step(step)

        # fourth step
        step_name = 'damped_adjustment_4'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-01_00:00:00'",
            'config_dt': "'00:03:00'",
            'config_btr_dt': "'00:00:06'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '4.0e-5',
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[2]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-01_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[2]}')
        step.add_output_file(filename=f'../{restart_filenames[3]}')
        self.add_step(step)

        # fifth step
        step_name = 'damped_adjustment_5'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-01_00:00:00'",
            'config_dt': "'00:04:00'",
            'config_btr_dt': "'00:00:08'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '2.0e-5',
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[3]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-01_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[3]}')
        step.add_output_file(filename=f'../{restart_filenames[4]}')
        self.add_step(step)

        # sixth step
        step_name = 'damped_adjustment_6'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-03_00:00:00'",
            'config_dt': "'00:04:00'",
            'config_btr_dt': "'00:00:08'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '5.0e-6',
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[4]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-03_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[4]}')
        step.add_output_file(filename=f'../{restart_filenames[5]}')
        self.add_step(step)

        # seventh step
        step_name = 'damped_adjustment_7'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-06_00:00:00'",
            'config_dt': "'00:05:00'",
            'config_btr_dt': "'00:00:10'",
            'config_implicit_bottom_drag_type': "'constant_and_rayleigh'",
            'config_Rayleigh_damping_coeff': '5.0e-6',
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[5]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-06_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[5]}')
        step.add_output_file(filename=f'../{restart_filenames[6]}')
        self.add_step(step)

        # final step
        step_name = 'simulation'
        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name, get_dt_from_min_res=False)

        namelist_options = {
            'config_run_duration': "'00-00-18_00:00:00'",
            'config_dt': "'00:05:00'",
            'config_btr_dt': "'00:00:10'",
            'config_do_restart': '.true.',
            'config_start_time': f"'{restart_times[6]}'"}
        namelist_options.update(shared_options)
        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-06_00:00:00'}
        step.add_streams_file(module, 'streams.template',
                              template_replacements=stream_replacements)

        step.add_input_file(filename=f'../{restart_filenames[6]}')
        step.add_output_file(filename=f'../{restart_filenames[7]}')
        step.add_output_file(filename='output.nc')
        self.add_step(step)

        self.restart_filenames = restart_filenames