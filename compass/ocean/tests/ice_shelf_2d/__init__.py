from compass.testgroup import TestGroup
from compass.ocean.tests.ice_shelf_2d.init import Init
from compass.ocean.tests.ice_shelf_2d.ssh_adjustment import SshAdjustment
from compass.ocean.tests.ice_shelf_2d.default import Default
from compass.ocean.tests.ice_shelf_2d.restart_test import RestartTest


class IceShelf2d(TestGroup):
    """
    A test group for ice-shelf 2D test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ice_shelf_2d')

        for resolution in ['5km']:
            for coord_type in ['z-star', 'z-level']:
                self.add_test_case(
                    Init(test_group=self, resolution=resolution,
                         coord_type=coord_type))
                self.add_test_case(
                    SshAdjustment(test_group=self, resolution=resolution,
                                  coord_type=coord_type))
                self.add_test_case(
                    Default(test_group=self, resolution=resolution,
                            coord_type=coord_type))
                self.add_test_case(
                    RestartTest(test_group=self, resolution=resolution,
                                coord_type=coord_type))
