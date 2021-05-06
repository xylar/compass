from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of ice-shelf 2D test
    cases.

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 run_duration=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        run_duration : str, optional
            The duration of the run
        """
        self.resolution = resolution
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=None, min_cores=None, threads=None)

        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward')

        if run_duration is not None:
            self.add_namelist_options(
                options={'config_run_duration': run_duration})

        self.add_streams_file('compass.ocean.streams',
                              'streams.land_ice_fluxes')

        self.add_streams_file('compass.ocean.tests.isomip_plus',
                              'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../ssh_adjustment/adjusted_init.nc')
        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file('output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)
