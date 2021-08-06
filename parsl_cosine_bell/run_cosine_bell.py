#!/usr/bin/env python
import os
import pickle
import configparser
import multiprocessing
import subprocess
import numpy as np
import xarray
import time
import matplotlib.pyplot as plt

from mpas_tools.logging import LoggingContext
import mpas_tools.io
from mpas_tools.logging import check_call
from mpas_tools.ocean import build_spherical_mesh


def main():

    with open('test_case.pickle', 'rb') as handle:
        test_case = pickle.load(handle)

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(test_case.config_filename)
    steps_to_run = config.get('test_case',
                              'steps_to_run').replace(',', ' ').split()

    base_dir = os.getcwd()

    for step_name in steps_to_run:
        step = test_case.steps[step_name]
        step_dir = os.path.join(base_dir, step.subdir)

        print(f'Running: {step_name}')
        run_step(step_dir)


# pulled out of Testcase.run() and its helper method Testcase._run_step()
def run_step(step_dir):
    """
    Run a step in its work directory

    Parameters
    ----------
    step_dir : str
        The work directory for the step
    """

    original_dir = os.getcwd()
    os.chdir(step_dir)

    with open('step.pickle', 'rb') as handle:
        test_case, step = pickle.load(handle)
    test_case.steps_to_run = [step.name]
    test_case.new_step_log_file = False

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(step.config_filename)
    test_case.config = config
    step.config = config

    mpas_tools.io.default_format = config.get('io', 'format')
    mpas_tools.io.default_engine = config.get('io', 'engine')

    # if this is a forward step, update the cores and min_cores from the
    # config file in case a user has changed them
    if 'forward' in step.name:
        resolution = step.resolution

        cores = config.getint('cosine_bell',
                              'QU{}_cores'.format(resolution))
        min_cores = config.getint('cosine_bell',
                                  'QU{}_min_cores'.format(resolution))
        step.cores = cores
        step.min_cores = min_cores

    # start logging to stdout/stderr
    test_name = step.path.replace('/', '_')
    with LoggingContext(name=test_name) as logger:
        test_case.logger = logger
        step.logger = logger

        available_cores, _ = get_available_cores_and_nodes(config)
        step.cores = min(step.cores, available_cores)
        if step.min_cores is not None:
            if step.cores < step.min_cores:
                raise ValueError(
                    'Available cores for {} is below the minimum of {}'
                    ''.format(step.cores, step.min_cores))

        missing_files = list()
        for input_file in step.inputs:
            if not os.path.exists(input_file):
                missing_files.append(input_file)

        if len(missing_files) > 0:
            raise OSError(
                'input file(s) missing in step {} of {}/{}/{}: {}'.format(
                    step.name, step.mpas_core.name, step.test_group.name,
                    step.test_case.subdir, missing_files))

        if 'mesh' in step.name:
            run_mesh(step)
        elif 'init' in step.name:
            run_init(step)
        elif 'forward' in step.name:
            run_forward(step)
        elif 'analysis' in step.name:
            run_analysis(step)
        else:
            raise ValueError(f'Unexpected step name: {step.name}')

        missing_files = list()
        for output_file in step.outputs:
            if not os.path.exists(output_file):
                missing_files.append(output_file)

        if len(missing_files) > 0:
            raise OSError(
                'output file(s) missing in step {} of {}/{}/{}: {}'.format(
                    step.name, step.mpas_core.name, step.test_group.name,
                    step.test_case.subdir, missing_files))

    os.chdir(original_dir)


# pulled out of the mesh step

def run_mesh(step):
    """
    Run this step of the test case
    """

    # create the base mesh
    cellWidth, lon, lat = build_cell_width_lat_lon(step)
    build_spherical_mesh(cellWidth, lon, lat, out_filename='mesh.nc',
                         logger=step.logger, use_progress_bar=False)

    make_graph_file(mesh_filename='mesh.nc',
                    graph_filename='graph.info')


def build_cell_width_lat_lon(step):
    """
    Create cell width array for this mesh on a regular latitude-longitude
    grid

    Returns
    -------
    cellWidth : numpy.array
        m x n array of cell width in km

    lon : numpy.array
        longitude in degrees (length n and between -180 and 180)

    lat : numpy.array
        longitude in degrees (length m and between -90 and 90)
    """
    dlon = 10.
    dlat = dlon
    constantCellWidth = float(step.resolution)

    nlat = int(180/dlat) + 1
    nlon = int(360/dlon) + 1
    lat = np.linspace(-90., 90., nlat)
    lon = np.linspace(-180., 180., nlon)

    cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
    return cellWidth, lon, lat


# from the init step

def run_init(step):
    run_model(step)


# from the forward step

def run_forward(step):
    """
    Run this step of the testcase
    """

    # update dt in case the user has changed dt_per_km
    dt = get_dt(step)
    update_namelist_at_runtime(step, options={'config_dt': dt},
                               out_name='namelist.ocean')

    run_model(step)


def get_dt(step):
    """
    Get the time step

    Returns
    -------
    dt : str
        the time step in HH:MM:SS
    """
    config = step.config
    # dt is proportional to resolution: default 30 seconds per km
    dt_per_km = config.getint('cosine_bell', 'dt_per_km')

    dt = dt_per_km * step.resolution
    # https://stackoverflow.com/a/1384565/7728169
    dt = time.strftime('%H:%M:%S', time.gmtime(dt))

    return dt


# from the analysis step

def run_analysis(step):
    """
    Run this step of the test case
    """
    plt.switch_backend('Agg')
    resolutions = step.resolutions
    xdata = list()
    ydata = list()
    for res in resolutions:
        rmseValue, nCells = rmse(step, res)
        xdata.append(nCells)
        ydata.append(rmseValue)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    p = np.polyfit(np.log10(xdata), np.log10(ydata), 1)
    conv = abs(p[0]) * 2.0

    yfit = xdata**p[0] * 10**p[1]

    plt.loglog(xdata, yfit, 'k')
    plt.loglog(xdata, ydata, 'or')
    plt.annotate('Order of Convergence = {}'.format(np.round(conv, 3)),
                 xycoords='axes fraction', xy=(0.3, 0.95), fontsize=14)
    plt.xlabel('Number of Grid Cells', fontsize=14)
    plt.ylabel('L2 Norm', fontsize=14)
    plt.savefig('convergence.png', bbox_inches='tight', pad_inches=0.1)


def rmse(step, resolution):
    """
    Compute the RMSE for a given resolution

    Parameters
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km

    Returns
    -------
    rmseValue : float
        The root-mean-squared error

    nCells : int
        The number of cells in the mesh
    """
    resTag = 'QU{}'.format(resolution)

    config = step.config
    latCent = config.getfloat('cosine_bell', 'lat_center')
    lonCent = config.getfloat('cosine_bell', 'lon_center')
    radius = config.getfloat('cosine_bell', 'radius')
    psi0 = config.getfloat('cosine_bell', 'psi0')
    pd = config.getfloat('cosine_bell', 'vel_pd')

    init = xarray.open_dataset('{}_init.nc'.format(resTag))
    # find time since the beginning of run
    ds = xarray.open_dataset('{}_output.nc'.format(resTag))
    for j in range(len(ds.xtime)):
        tt = str(ds.xtime[j].values)
        tt.rfind('_')
        DY = float(tt[10:12]) - 1
        if DY == pd:
            sliceTime = j
            break
    HR = float(tt[13:15])
    MN = float(tt[16:18])
    t = 86400.0 * DY + HR * 3600. + MN
    # find new location of blob center
    # center is based on equatorial velocity
    R = init.sphere_radius
    distTrav = 2.0 * 3.14159265 * R / (86400.0 * pd) * t
    # distance in radians is
    distRad = distTrav / R
    newLon = lonCent + distRad
    if newLon > 2.0 * np.pi:
        newLon -= 2.0 * np.pi

    # construct analytic tracer
    tracer = np.zeros_like(init.tracer1[0, :, 0].values)
    latC = init.latCell.values
    lonC = init.lonCell.values
    temp = R * np.arccos(np.sin(latCent) * np.sin(latC) +
                         np.cos(latCent) * np.cos(latC) * np.cos(
        lonC - newLon))
    mask = temp < radius
    tracer[mask] = psi0 / 2.0 * (
                1.0 + np.cos(3.1415926 * temp[mask] / radius))

    # oad forward mode data
    tracerF = ds.tracer1[sliceTime, :, 0].values
    rmseValue = np.sqrt(np.mean((tracerF - tracer)**2))

    init.close()
    ds.close()
    return rmseValue, init.dims['nCells']


# some helper functions pulled out of the compass framework

def update_namelist_at_runtime(step, options, out_name=None):
    """
    Update an existing namelist file with additional options.  This would
    typically be used for namelist options that are only known at runtime,
    not during setup, typically those related to the number of nodes and
    cores.

    Parameters
    ----------
    options : dict
        A dictionary of options and value to replace namelist options with
        new values

    out_name : str, optional
        The name of the namelist file to write out, ``namelist.<core>`` by
        default
    """

    if out_name is None:
        out_name = 'namelist.{}'.format(step.mpas_core.name)

    filename = '{}/{}'.format(step.work_dir, out_name)

    namelist = namelist_ingest(filename)

    namelist = namelist_replace(namelist, options)

    namelist_write(namelist, filename)


def namelist_ingest(defaults_filename):
    """ Read the defaults file """
    with open(defaults_filename, 'r') as f:
        lines = f.readlines()

    namelist = dict()
    record = None
    for line in lines:
        if '&' in line:
            record = line.strip('&').strip('\n').strip()
            namelist[record] = dict()
        elif '=' in line:
            if record is not None:
                opt, val = line.strip('\n').split('=')
                namelist[record][opt.strip()] = val.strip()

    return namelist


def namelist_replace(namelist, replacements):
    """ Replace entries in the namelist using the replacements dict """
    new = dict(namelist)
    for record in new:
        for key in replacements:
            if key in new[record]:
                new[record][key] = replacements[key]

    return new


def namelist_write(namelist, filename):
    """ Write the namelist out """

    with open(filename, 'w') as f:
        for record in namelist:
            f.write('&{}\n'.format(record))
            rec = namelist[record]
            for key in rec:
                f.write('    {} = {}\n'.format(key.strip(), rec[key].strip()))
            f.write('/\n')


def run_model(step, update_pio=True, partition_graph=True,
              graph_file='graph.info', namelist=None, streams=None):
    """
    Run the model after determining the number of cores

    Parameters
    ----------
    step : compass.Step
        a step

    update_pio : bool, optional
        Whether to modify the namelist so the number of PIO tasks and the
        stride between them is consistent with the number of nodes and cores
        (one PIO task per node).

    partition_graph : bool, optional
        Whether to partition the domain for the requested number of cores.  If
        so, the partitioning executable is taken from the ``partition`` option
        of the ``[executables]`` config section.

    graph_file : str, optional
        The name of the graph file to partition

    namelist : str, optional
        The name of the namelist file, default is ``namelist.<core>``

    streams : str, optional
        The name of the streams file, default is ``streams.<core>``
    """
    mpas_core = step.mpas_core.name
    cores = step.cores
    threads = step.threads
    config = step.config
    logger = step.logger

    if namelist is None:
        namelist = 'namelist.{}'.format(mpas_core)

    if streams is None:
        streams = 'streams.{}'.format(mpas_core)

    if update_pio:
        step.update_namelist_pio(namelist)

    if partition_graph:
        partition(cores, config, logger, graph_file=graph_file)

    os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)

    parallel_executable = config.get('parallel', 'parallel_executable')
    model = config.get('executables', 'model')
    model_basename = os.path.basename(model)

    # split the parallel executable into constituents in case it includes flags
    args = parallel_executable.split(' ')
    args.extend(['-n', '{}'.format(cores),
                 './{}'.format(model_basename),
                 '-n', namelist,
                 '-s', streams])

    check_call(args, logger)


def partition(cores, config, logger, graph_file='graph.info'):
    """
    Partition the domain for the requested number of cores

    Parameters
    ----------
    cores : int
        The number of cores that the model should be run on

    config : configparser.ConfigParser
        Configuration options for the test case, used to get the partitioning
        executable

    logger : logging.Logger
        A logger for output from the step that is calling this function

    graph_file : str, optional
        The name of the graph file to partition

    """
    if cores > 1:
        executable = config.get('parallel', 'partition_executable')
        args = [executable, graph_file, '{}'.format(cores)]
        check_call(args, logger)


def make_graph_file(mesh_filename, graph_filename='graph.info',
                    weight_field=None):
    """
    Make a graph file from the MPAS mesh for use in the Metis graph
    partitioning software

    Parameters
    ----------
     mesh_filename : str
        The name of the input MPAS mesh file

    graph_filename : str, optional
        The name of the output graph file

    weight_field : str
        The name of a variable in the MPAS mesh file to use as a field of
        weights
    """

    with xarray.open_dataset(mesh_filename) as ds:

        nCells = ds.sizes['nCells']

        nEdgesOnCell = ds.nEdgesOnCell.values
        cellsOnCell = ds.cellsOnCell.values - 1
        if weight_field is not None:
            if weight_field in ds:
                raise ValueError('weight_field {} not found in {}'.format(
                    weight_field, mesh_filename))
            weights = ds[weight_field].values
        else:
            weights = None

    nEdges = 0
    for i in range(nCells):
        for j in range(nEdgesOnCell[i]):
            if cellsOnCell[i][j] != -1:
                nEdges = nEdges + 1

    nEdges = nEdges/2

    with open(graph_filename, 'w+') as graph:
        if weights is None:
            graph.write('{} {}\n'.format(nCells, nEdges))

            for i in range(nCells):
                for j in range(0, nEdgesOnCell[i]):
                    if cellsOnCell[i][j] >= 0:
                        graph.write('{} '.format(cellsOnCell[i][j]+1))
                graph.write('\n')
        else:
            graph.write('{} {} 010\n'.format(nCells, nEdges))

            for i in range(nCells):
                graph.write('{} '.format(int(weights[i])))
                for j in range(0, nEdgesOnCell[i]):
                    if cellsOnCell[i][j] >= 0:
                        graph.write('{} '.format(cellsOnCell[i][j] + 1))
                graph.write('\n')


def get_available_cores_and_nodes(config):
    """
    Get the number of total cores and nodes available for running steps

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for the test case

    Returns
    -------
    cores : int
        The number of cores available for running steps

    nodes : int
        The number of cores available for running steps
    """

    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm':
        job_id = os.environ['SLURM_JOB_ID']
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%C']
        cores = _get_subprocess_int(args)
        args = ['squeue', '--noheader', '-j', job_id, '-o', '%D']
        nodes = _get_subprocess_int(args)
    elif parallel_system == 'single_node':
        cores_per_node = config.getint('parallel', 'cores_per_node')
        cores = min(multiprocessing.cpu_count(), cores_per_node)
        nodes = 1
    else:
        raise ValueError('Unexpected parallel system: {}'.format(
            parallel_system))

    return cores, nodes


def _get_subprocess_int(args):
    value = subprocess.check_output(args)
    value = int(value.decode('utf-8').strip('\n'))
    return value


if __name__ == '__main__':
    main()
