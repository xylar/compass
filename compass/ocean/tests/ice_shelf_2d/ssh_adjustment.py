from compass.ocean.iceshelf.ssh_adjustment import SshAdjustment as \
    SshAdjustmentBase


class SshAdjustment(SshAdjustmentBase):
    """
    Set up the initial condition for other ice-shelf 2D test cases
    """

    def __init__(self, test_group, resolution, coord_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.ice_shelf_2d.IceShelf2d
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case

        coord_type : str
            The type of vertical coordinate (``z-star``, ``z-level``, etc.)
        """
        name = 'ssh_adjustment'
        subdir = f'{resolution}/{coord_type}/{name}'
        init_dir = '../../init/initial_state'
        super().__init__(test_group=test_group, name=name, subdir=subdir,
                         variable='landIcePressure',
                         init_target_filename=f'{init_dir}/initial_state.nc',
                         graph_target_filename=f'{init_dir}/culled_graph.info')

    def configure(self):
        """
        Add the steps based on the ``iteration_count`` config option
        """
        super().configure()

        for step_name, step in self.steps.items():
            if step_name.startswith('forward'):
                step.add_namelist_file('compass.ocean.tests.ice_shelf_2d',
                                       'namelist.forward')

