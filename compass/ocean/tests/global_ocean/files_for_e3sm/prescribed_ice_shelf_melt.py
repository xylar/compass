import os
import xarray

from compass.io import symlink, add_input_file
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path


def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    defaults = dict(cores=1, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    add_input_file(step, filename='README', target='../README')
    add_input_file(step, filename='restart.nc',
                   target='../{}'.format(step['restart_filename']))
    prescribed_ismf_path = \
        '{}/prescribed_ice_shelf_melt/remap_ice_shelf_melt'.format(
            get_mesh_relative_path(step))

    add_input_file(step, filename='prescribed_ismf_rignot2013.nc',
                   target='{}/prescribed_ismf_rignot2013.nc'.format(
                       prescribed_ismf_path))

    # for now, we won't define any outputs because they include the mesh short
    # name, which is not known at setup time.  Currently, this is safe because
    # no other steps depend on the outputs of this one.


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step
    """
    restart_filename = 'restart.nc'

    with xarray.open_dataset(restart_filename) as ds:
        mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
        mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
        prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)
        creation_date = ds.attrs['{}_Version_Creation_Date'.format(prefix)]

    directory = '../assembled_files/inputdata/ocn/mpas-o/{}'.format(
        mesh_short_name)
    try:
        os.makedirs(directory)
    except OSError:
        pass

    in_filename = 'prescribed_ismf_rignot2013.nc'

    out_filename = 'prescribed_ismf_rignot2013.{}.{}.nc'.format(
        mesh_short_name, creation_date)

    symlink('../../../../../prescribed_ice_shelf_melt/{}'.format(in_filename),
            '{}/{}'.format(directory, out_filename))
