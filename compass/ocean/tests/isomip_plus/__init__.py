from compass.testgroup import TestGroup

from compass.ocean.tests.isomip_plus.ocean_test import OceanTest


class IsomipPlus(TestGroup):
    """
    A test group for ice-shelf 2D test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='isomip_plus')

        for resolution in [2., 5.]:
            for experiment in ['Ocean0']:
                for vertical_coordinate in ['z-star']:
                    self.add_test_case(
                        OceanTest(test_group=self, resolution=resolution,
                                  experiment=experiment,
                                  vertical_coordinate=vertical_coordinate))


def configure(resolution, config):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    config : configparser.ConfigParser
        Configuration options for this test case
    """

    nx = round(800/resolution)
    ny = round(100/resolution)
    dc = 1e3*resolution

    config.set('isomip_plus', 'nx', '{}'.format(nx))
    config.set('isomip_plus', 'ny', '{}'.format(ny))
    config.set('isomip_plus', 'dc', '{}'.format(dc))
