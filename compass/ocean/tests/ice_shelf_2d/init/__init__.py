from compass.testcase import TestCase
from compass.ocean.tests.ice_shelf_2d.init.initial_state import InitialState
from compass.validate import compare_variables


class Init(TestCase):
    """
    Set up the initial condition for other ice-shelf 2D test cases

    Attributes
    ----------
    resolution : str
        The horizontal resolution of the test case

    coord_type : str
        The type of vertical coordinate (``z-star``, ``z-level``, etc.)
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
        name = 'init'
        self.resolution = resolution
        self.coord_type = coord_type
        subdir = f'{resolution}/{coord_type}/{name}'
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(InitialState(test_case=self, resolution=resolution))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        res_params = {'5km': {'nx': 10, 'ny': 44, 'dc': 5e3}}
        resolution = self.resolution
        config = self.config
        coord_type = self.coord_type

        if resolution not in res_params:
            raise ValueError(
                f'Unsupported resolution {resolution}. Supported values are: '
                f'{list(res_params)}')
        res_params = res_params[resolution]
        for param in res_params:
            config.set('ice_shelf_2d', param, f'{res_params[param]}')

        config.set('vertical_grid', 'coord_type', coord_type)
        if coord_type == 'z-level':
            # we need more vertical resolution
            config.set('vertical_grid', 'vert_levels', '100')

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['bottomDepth', 'ssh', 'layerThickness', 'zMid',
                     'maxLevelCell', 'temperature', 'salinity']
        compare_variables(
            test_case=self, variables=variables,
            filename1='initial_state/initial_state.nc')
