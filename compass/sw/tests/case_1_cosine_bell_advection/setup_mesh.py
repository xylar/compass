import numpy
from netCDF4 import Dataset as NetCDFFile

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.logging import check_call

from compass.io import add_input_file, add_output_file
from compass.model import make_graph_file


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
    mesh_type = step['mesh_type']

    defaults = dict(cores=1, min_cores=1, max_memory=8000, max_disk=8000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    if mesh_type == 'variable_resolution':
        # download and link the mesh

        add_input_file(step, filename='mpas_grid.nc',
                       target='case_1_cosine_bell_advection_varres_grid.nc', database='')

    add_output_file(step, filename='graph.info')
    add_output_file(step, filename='sw_grid.nc')


# no setup function is needed


def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
   """
    mesh_type = step['mesh_type']
    section = config['case_1_cosine_bell_advection']

    if mesh_type == '2000m':
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
                                      nonperiodic_y=True)

        write_netcdf(dsMesh, 'grid.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'mpas_grid.nc')

    levels = section.getfloat('levels')
    args = ['create_sw_grid_from_generic_MPAS_grid.py',
            '-i', 'mpas_grid.nc',
            '-o', 'sw_grid.nc',
            '-l', levels]

    check_call(args, logger)

    make_graph_file(mesh_filename='sw_grid.nc',
                    graph_filename='graph.info')

    _setup_case_1_cosine_bell_advection_initial_conditions(config, logger, filename='sw_grid.nc')


def _setup_case_1_cosine_bell_advection_initial_conditions(config, logger, filename='sw_grid.nc'):
    """
    Add the initial condition to the given MPAS mesh file

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str, optional
        file to setup case_1_cosine_bell_advection
    """
    section = config['case_1_cosine_bell_advection']
    case_1_cosine_bell_advection_type = section.get('case_1_cosine_bell_advection_type')
    put_origin_on_a_cell = section.getboolean('put_origin_on_a_cell')
    shelf = section.getboolean('shelf')
    hydro = section.getboolean('hyrdo')

    # Open the file, get needed dimensions
    gridfile = NetCDFFile(filename, 'r+')
    nVertLevels = len(gridfile.dimensions['nVertLevels'])
    # Get variables
    xCell = gridfile.variables['xCell']
    yCell = gridfile.variables['yCell']
    xEdge = gridfile.variables['xEdge']
    yEdge = gridfile.variables['yEdge']
    xVertex = gridfile.variables['xVertex']
    yVertex = gridfile.variables['yVertex']
    thickness = gridfile.variables['thickness']
    bedTopography = gridfile.variables['bedTopography']
    layerThicknessFractions = gridfile.variables['layerThicknessFractions']
    SMB = gridfile.variables['sfcMassBal']

    # Find center of domain
    x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
    y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
    # Calculate distance of each cell center from case_1_cosine_bell_advection center
    r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    if put_origin_on_a_cell:
        # Center the case_1_cosine_bell_advection in the center of the cell that is closest to the center
        # of the domain.
        centerCellIndex = numpy.abs(r[:]).argmin()
        xShift = -1.0 * xCell[centerCellIndex]
        yShift = -1.0 * yCell[centerCellIndex]
        xCell[:] = xCell[:] + xShift
        yCell[:] = yCell[:] + yShift
        xEdge[:] = xEdge[:] + xShift
        yEdge[:] = yEdge[:] + yShift
        xVertex[:] = xVertex[:] + xShift
        yVertex[:] = yVertex[:] + yShift
        # Now update origin location and distance array
        x0 = 0.0
        y0 = 0.0
        r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    # Assign variable values for case_1_cosine_bell_advection
    # Define case_1_cosine_bell_advection dimensions - all in meters
    r0 = 60000.0 * numpy.sqrt(0.125)
    h0 = 2000.0 * numpy.sqrt(0.125)
    # Set default value for non-case_1_cosine_bell_advection cells
    thickness[:] = 0.0
    # Calculate the case_1_cosine_bell_advection thickness for cells within the desired radius
    # (thickness will be NaN otherwise)
    thickness_field = thickness[0, :]
    if case_1_cosine_bell_advection_type == 'cism':
        thickness_field[r < r0] = h0 * (1.0 - (r[r < r0] / r0) ** 2) ** 0.5
    elif case_1_cosine_bell_advection_type == 'halfar':
        thickness_field[r < r0] = h0 * (
                    1.0 - (r[r < r0] / r0) ** (4.0 / 3.0)) ** (3.0 / 7.0)
    else:
        raise ValueError('Unexpected case_1_cosine_bell_advection_type: {}'.format(case_1_cosine_bell_advection_type))
    thickness[0, :] = thickness_field

    # zero velocity everywhere
    # normalVelocity[:] = 0.0
    # flat bed at sea level
    bedTopography[:] = 0.0
    if shelf:
        # this line will make a small shelf:
        bedTopography[0, xCell[:] < -10000.0] = -2000.0
    # Setup layerThicknessFractions
    layerThicknessFractions[:] = 1.0 / nVertLevels

    # boundary conditions
    # Sample values to use, or comment these out for them to be 0.
    SMB[:] = 0.0
    # beta[:] = 50000.
    # units: m/yr, lapse rate of 1 m/yr with 0 at 500 m
    # SMB[:] = 2.0/1000.0 * (thickness[:] + bedTopography[:]) - 1.0
    # Convert from units of m/yr to kg/m2/s using an assumed ice density
    SMB[:] = SMB[:] * 910.0 / (3600.0 * 24.0 * 365.0)

    # lapse rate of 5 deg / km
    # Tsfc[:, 0] = -5.0/1000.0 * (thickness[0,:] + bedTopography[0,:])
    # G = 0.01
    # BMB[:] = -20.0  # units: m/yr

    if hydro:
        gridfile.variables['uReconstructX'][:] = 5.0 / (3600.0 * 24.0 * 365.0)
        gridfile.variables['basalMeltInput'][:] = 0.06 / 335000.0 * 50.0
        gridfile.variables['externalWaterInput'][:] = 0.0
        gridfile.variables['waterThickness'][:] = 0.08

    gridfile.close()

    logger.info('Successfully added case_1_cosine_bell_advection initial conditions to: {}'.format(
        filename))
