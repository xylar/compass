from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.namelist import add_namelist_file
from compass.streams import add_streams_file
from compass.io import add_output_file
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.subdir import get_forward_sudbdir
from compass.ocean.tests.global_ocean.prescribed_ice_shelf_melt import \
    remap_ice_shelf_melt


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    mesh_name = testcase['mesh_name']
    initial_condition = testcase['initial_condition']
    with_bgc = testcase['with_bgc']
    time_integrator = testcase['time_integrator']
    name = testcase['name']

    testcase['description'] = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='prescribed ice-shelf melt')

    subdir = get_forward_sudbdir(mesh_name, initial_condition, with_bgc,
                                 time_integrator, name)
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, remap_ice_shelf_melt, mesh_name=mesh_name)

    step = add_step(testcase, forward, mesh_name=mesh_name,
                    with_ice_shelf_cavities=True,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

    module = __name__
    add_namelist_file(step, module, 'namelist.forward')
    add_streams_file(step, module, 'streams.forward')
    add_output_file(step, filename='land_ice_fluxes.nc')


def run(testcase, test_suite, config, logger):
    """
    Run each step of the testcase

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the test case
    """
    run_steps(testcase, test_suite, config, logger)
