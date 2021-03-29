import os
import xarray
import numpy
import pyproj

from pyremap import MpasMeshDescriptor, ProjectionGridDescriptor, Remapper
from mpas_tools.io import write_netcdf
from mpas_tools.cime.constants import constants

from compass.io import add_input_file, add_output_file
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
    defaults = dict(cores=36, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    # this file isn't publicly available and has to be added manually by those
    # who have permission to access it
    add_input_file(
        step, filename='MeltRatesRignot2013.nc',
        target='MeltRatesRignot2013.nc',
        database='initial_condition_database')

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))
    add_input_file(step, filename='mesh.nc',
                   target='{}/culled_mesh.nc'.format(mesh_path))

    add_output_file(step, filename='prescribed_ismf_rignot2013.nc')


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
    mesh_filename = 'mesh.nc'
    mesh_name = step['mesh_name']
    cores = step['cores']

    in_filename = 'MeltRatesRignot2013.nc'

    out_filename = 'prescribed_ismf_rignot2013.nc'

    if 'ESMF' in os.environ:
        parallel_executable = config.get('parallel', 'parallel_executable')
        esmf_path = os.environ['ESMF']
    else:
        parallel_executable = None
        esmf_path = None

    remap_rignot(in_filename, mesh_filename, mesh_name, out_filename,
                 logger=logger,  mpi_tasks=cores,
                 parallel_executable=parallel_executable, esmf_path=esmf_path)


def remap_rignot(in_filename, mesh_filename, mesh_name, out_filename,
                 logger, mapping_directory='.', method='conserve',
                 renormalization_threshold=None, in_var_name='melt_actual',
                 mpi_tasks=1, parallel_executable=None, esmf_path=None):
    """
    Remap the Rignot et al. (2013) melt rates at 1 km resolution to an MPAS
    mesh

    Parameters
    ----------
    in_filename : str
        The original Rignot et al. (2013) melt rates

    mesh_filename : str
        The MPAS mesh

    mesh_name : str
        The name of the mesh (e.g. oEC60to30wISC), used in the name of the
        mapping file

    out_filename : str
        The melt rates interpolated to the MPAS mesh with ocean sensible heat
        fluxes added on (assuming insulating ice)

    logger : logging.Logger
        A logger for output from the step

    mapping_directory : str
        The directory where the mapping file should be stored (if it is to be
        computed) or where it already exists (if not)

    method : {'bilinear', 'neareststod', 'conserve'}, optional
        The method of interpolation used, see documentation for
        `ESMF_RegridWeightGen` for details.

    renormalization_threshold : float, optional
        The minimum weight of a denstination cell after remapping, below
        which it is masked out, or ``None`` for no renormalization and
        masking.

    in_var_name : {'melt_actual', 'melt_steadystate'}
        Whether to use the melt rate for the time period covered in Rignot et
        al. (2013) with observed thinning/thickening or the melt rates that
        would be required if ice shelves were in steady state.

    mpi_tasks : int, optional
        The number of MPI tasks to use to compute the mapping file

    esmf_path : str, optional
        A path to a system build of ESMF (containing a 'bin' directory with
        the ESMF tools).  By default, ESMF tools are found in the conda
        environment

    parallel_executable : {'srun', 'mpirun'}, optional
        The name of the parallel executable to use to launch ESMF tools.
        But default, 'mpirun' from the conda environment is used
    """

    ds = xarray.open_dataset(in_filename)
    lx = numpy.abs(1e-3 * (ds.xaxis.values[-1] - ds.xaxis.values[0]))
    ly = numpy.abs(1e-3 * (ds.yaxis.values[-1] - ds.yaxis.values[0]))

    inGridName = '{}x{}km_1.0km_Antarctic_stereo'.format(lx, ly)

    projection = pyproj.Proj('+proj=stere +lat_ts=-71.0 +lat_0=-90 +lon_0=0.0 '
                             '+k_0=1.0 +x_0=0.0 +y_0=0.0 +ellps=WGS84')

    inDescriptor = ProjectionGridDescriptor.read(
        projection,  in_filename, xVarName='xaxis', yVarName='yaxis',
        meshName=inGridName)

    # convert to the units and variable names expected in MPAS-O
    rho_fw = constants['SHR_CONST_RHOFW']
    s_per_yr = 365.*constants['SHR_CONST_CDAY']
    latent_heat_of_fusion = constants['SHR_CONST_LATICE']
    ds['prescribedLandIceFreshwaterFlux'] = ds[in_var_name]*rho_fw/s_per_yr
    ds['prescribedLandIceHeatFlux'] = (latent_heat_of_fusion *
                                       ds['prescribedLandIceFreshwaterFlux'])
    ds = ds.drop_vars(['melt_actual', 'melt_steadystate', 'lon', 'lat'])

    outDescriptor = MpasMeshDescriptor(mesh_filename, mesh_name)

    mappingFileName = '{}/map_{}_to_{}.nc'.format(
        mapping_directory, inGridName, mesh_name)

    remapper = Remapper(inDescriptor, outDescriptor, mappingFileName)

    remapper.build_mapping_file(method=method, mpiTasks=mpi_tasks,
                                tempdir=mapping_directory, logger=logger,
                                esmf_path=esmf_path,
                                esmf_parallel_exec=parallel_executable)

    dsRemap = remapper.remap(
        ds, renormalizationThreshold=renormalization_threshold)

    for field in ['prescribedLandIceFreshwaterFlux',
                  'prescribedLandIceHeatFlux']:
        # zero out the field where it's currently NaN
        dsRemap[field] = dsRemap[field].where(dsRemap[field].notnull(), 0.)

    dsRemap.attrs.pop('history')

    write_netcdf(dsRemap, out_filename)
