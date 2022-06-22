from compass.testcase import TestCase
from compass.ocean.tests.ice_shelf_2d.forward import Forward
from compass.ocean.tests.ice_shelf_2d.viz import Viz
from compass.validate import compare_variables


class RestartTest(TestCase):
    """
    A restart test case for the ice-shelf 2D test case test group, which makes
    sure the model produces identical results with one longer run and two
    shorter runs with a restart in between.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

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
        name = 'restart_test'
        self.resolution = resolution
        self.coord_type = coord_type
        subdir = f'{resolution}/{coord_type}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        for part in ['full', 'restart']:
            name = '{}_run'.format(part)
            step = Forward(test_case=self, name=name, subdir=name, ntasks=4,
                           openmp_threads=1, resolution=resolution,
                           with_frazil=True)

            step.add_namelist_file(
                'compass.ocean.tests.ice_shelf_2d.restart_test',
                'namelist.{}'.format(part))
            step.add_streams_file(
                'compass.ocean.tests.ice_shelf_2d.restart_test',
                'streams.{}'.format(part))
            self.add_step(step)

        self.add_step(Viz(test_case=self), run_by_default=False)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')

        variables = ['ssh', 'landIcePressure', 'landIceDraft',
                     'landIceFraction',
                     'landIceMask', 'landIceFrictionVelocity', 'topDrag',
                     'topDragMagnitude', 'landIceFreshwaterFlux',
                     'landIceHeatFlux', 'heatFluxToLandIce',
                     'landIceBoundaryLayerTemperature',
                     'landIceBoundaryLayerSalinity',
                     'landIceHeatTransferVelocity',
                     'landIceSaltTransferVelocity',
                     'landIceInterfaceTemperature',
                     'landIceInterfaceSalinity', 'accumulatedLandIceMass',
                     'accumulatedLandIceHeat']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/land_ice_fluxes.nc',
                          filename2='restart_run/land_ice_fluxes.nc')

        variables = ['accumulatedFrazilIceMass',
                     'accumulatedFrazilIceSalinity',
                     'seaIceEnergy', 'frazilLayerThicknessTendency',
                     'frazilTemperatureTendency', 'frazilSalinityTendency',
                     'frazilSurfacePressure',
                     'accumulatedLandIceFrazilMass']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/frazil.nc',
                          filename2='restart_run/frazil.nc')
