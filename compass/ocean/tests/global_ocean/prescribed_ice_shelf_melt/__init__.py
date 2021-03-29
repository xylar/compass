from compass.testcase import set_testcase_subdir, add_step, run_steps
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
    name = testcase['name']

    testcase['description'] = \
        'global ocean {} prescribed ice-shelf melt'.format(mesh_name)

    subdir = '{}/{}'.format(mesh_name, name)
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, remap_ice_shelf_melt, mesh_name=mesh_name)


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
