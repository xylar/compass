from compass.testcase import TestCase
from compass.ocean.tests.isomip_plus.initial_state import InitialState
from compass.ocean.tests.isomip_plus.ssh_adjustment import SshAdjustment
from compass.ocean.tests.isomip_plus.forward import Forward
from compass.ocean.tests import isomip_plus
from compass.validate import compare_variables


class OceanTest(TestCase):
    """
    An ISOMIP+ test case

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment

    vertical_coordinate : {'z-star', 'z-level', 'haney-number'}
        The vertical coordinate
    """

    def __init__(self, test_group, resolution, experiment,
                 vertical_coordinate):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.isomip_plus.IsomipPlus
            The test group that this test case belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment

        vertical_coordinate : {'z-star', 'z-level', 'haney-number'}
            The vertical coordinate
        """
        name = experiment
        self.resolution = resolution
        self.experiment = experiment
        self.vertical_coordinate = vertical_coordinate

        if resolution == int(resolution):
            res_folder = '{}km'.format(int(resolution))
        else:
            res_folder = '{}km'.format(resolution)

        subdir = '{}/{}/{}'.format(res_folder, vertical_coordinate, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution,
                         experiment=experiment,
                         vertical_coordinate=vertical_coordinate))
        self.add_step(
            SshAdjustment(test_case=self))
        self.add_step(
            Forward(test_case=self, name='performance', resolution=resolution,
                    run_duration='0000-00-00_01:00:00'))

        self.add_step(
            Forward(test_case=self, name='simulation', resolution=resolution,
                    run_duration='0000-01-00_00:00:00'),
            run_by_default=False)

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        isomip_plus.configure(self.resolution, self.config)

    def run(self):
        """
        Run each step of the test case
        """
        config = self.config
        # get the these properties from the config options
        for step_name in self.steps_to_run:
            if step_name in ['ssh_adjustment', 'performance', 'simulation']:
                step = self.steps[step_name]
                # get the these properties from the config options
                step.cores = config.getint('isomip_plus', 'forward_cores')
                step.min_cores = config.getint('isomip_plus',
                                               'forward_min_cores')
                step.threads = config.getint('isomip_plus', 'forward_threads')

        # run the steps
        super().run()

        # perform validation
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')

        variables = \
            ['ssh', 'landIcePressure', 'landIceDraft', 'landIceFraction',
             'landIceMask', 'landIceFrictionVelocity', 'topDrag',
             'topDragMagnitude', 'landIceFreshwaterFlux',
             'landIceHeatFlux', 'heatFluxToLandIce',
             'landIceBoundaryLayerTemperature', 'landIceBoundaryLayerSalinity',
             'landIceHeatTransferVelocity', 'landIceSaltTransferVelocity',
             'landIceInterfaceTemperature', 'landIceInterfaceSalinity',
             'accumulatedLandIceMass', 'accumulatedLandIceHeat']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/land_ice_fluxes.nc')
