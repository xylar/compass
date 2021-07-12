#!/usr/bin/env python

import os
import numpy as np
import xarray
import subprocess
import matplotlib.pyplot as plt
from mpas_tools.io import write_netcdf
from mpas_tools.ocean import build_spherical_mesh
from mpas_tools.cime.constants import constants
from mpas_tools.viz.paraview_extractor import extract_vtk
from pyremap.descriptor import LatLonGridDescriptor, MpasMeshDescriptor
from compass.io import symlink


def build_lat_lon_grid(dlon=1., dlat=1.):
    """
    Define a longitude and latitude grid with the given resolution

    Parameters
    ----------
    dlon : float
        The resolution in degrees of the longitude coordinate

    dlat : float
        The resolution in degrees of the latitude coordinate

    Returns
    -------
    lon : numpy.ndarray
        longitude in degrees (length n and between -180 and 180)

    lat : numpy.ndarray
        longitude in degrees (length m and between -90 and 90)
    """
    nlat = int(180/dlat) + 1
    nlon = int(360/dlon) + 1
    lat = np.linspace(-90., 90., nlat)
    lon = np.linspace(-180., 180., nlon)

    return lon, lat


def build_cell_width_lat_lon(north_res, south_res):
    """
    Define cell widths in km on a regular latitude-longitude grid, varying
    linearly from one resolution at the North Pole to another at the South
    Pole

    Parameters
    ----------
    north_res : float
        The resolution in km of the mesh at the North Pole

    south_res : float
        The resolution in km of the mesh at the South Pole

    Returns
    -------
    cellWidth : numpy.ndarray
        m x n array of cell width in km

    lon : numpy.ndarray
        longitude in degrees (length n and between -180 and 180)

    lat : numpy.ndarray
        longitude in degrees (length m and between -90 and 90)
    """
    lon, lat = build_lat_lon_grid(dlon=1., dlat=1.)

    nlat = lat.shape[0]

    cell_width = np.linspace(south_res, north_res, nlat)
    # broadcast cell_widths from 1D to 2D
    _, cell_width = np.meshgrid(lon, cell_width)

    return cell_width, lon, lat


def build_cosine_bell_map(filename, lon_center=180., lat_center=0., psi0=1.,
                          radius=2123666.667):
    """
    Writes out the cosine-bell field on longitude-latitude grid
    Pole

    Parameters
    ----------
    filename : str
        The file to write to

    lon_center : float, optional
        The longitude location of the center of the cosine bell in degrees

    lat_center : float, optional
        The latitude location of the center of the cosine bell in degrees

    psi0 : float, optional
        Hill max of cosine bell

    radius : float, optional
        Radius of cosine bell

    """
    lon, lat = build_lat_lon_grid(dlon=1., dlat=1.)

    Lon, Lat = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))

    earth_radius = constants['SHR_CONST_REARTH']

    lat_center = np.deg2rad(lat_center)
    lon_center = np.deg2rad(lon_center)

    distance = earth_radius * np.arccos(
        np.sin(lat_center) * np.sin(Lat) +
        np.cos(lat_center) * np.cos(Lat) * np.cos(Lon - lon_center))

    mask = distance < radius

    cosine_bell = np.zeros(Lon.shape)
    cosine_bell[mask] = psi0/2. * ( 1. + np.cos(np.pi * distance[mask]/radius))

    da = xarray.DataArray(data=cosine_bell, dims=('lat', 'lon'),
                          coords={'lon': lon, 'lat': lat})
    da.lon.attrs['units'] = 'degrees'
    da.lat.attrs['units'] = 'degrees'
    ds = xarray.Dataset()
    ds['cosineBell'] = da

    write_netcdf(ds, fileName=filename)


def build_mesh(north_res, south_res, mesh_filename):
    """
    Build an MPAS mesh.

    Parameters
    ----------
    north_res : float
        The resolution of the mesh in km at the North Pole

    south_res : float
        The resolution of the mesh in km at the South Pole

    mesh_filename : str
        The name of the file to write the MPAS mesh to
    """
    cell_width, lon, lat = build_cell_width_lat_lon(north_res, south_res)
    build_spherical_mesh(cell_width, lon, lat, out_filename=mesh_filename,
                         vtk_dir='vtk_mesh_lon_lat')

    extract_vtk(ignore_time=True, dimension_list=['maxEdges='],
                variable_list=['allOnCells'], filename_pattern=mesh_filename,
                out_dir='vtk_mesh_sphere')


def write_scrip_files(lon_lat_filename, mpas_mesh_filename, mpas_mesh_name,
                      src_scrip_filename, dst_scrip_filename):
    """
    Write SCRIP files that describe the lon/lat grid and MPAS mesh

    Parameters
    ----------
    lon_lat_filename : str
        The name of the file containing lon/lat grid

    mpas_mesh_filename: str
        The name of the file containing the MPAS mesh

    mpas_mesh_name : str
        The name of the MPAS mesh

    src_scrip_filename : str
        The name of the SCRIP file to write for the lon/lat grid

    dst_scrip_filename : str
        The name of the SCRIP file to write for the MPAS mesh

    Returns
    -------
    src_descriptor : LatLonGridDescriptor
        An object that describes the lon/lat grid

    dst_descriptor : MpasMeshDescriptor
        An object that describes the MPAS mesh
    """
    src_descriptor = LatLonGridDescriptor.read(lon_lat_filename)
    src_descriptor.to_scrip(src_scrip_filename)

    dst_descriptor = MpasMeshDescriptor(mpas_mesh_filename,
                                        meshName=mpas_mesh_name)
    dst_descriptor.to_scrip(dst_scrip_filename)

    return src_descriptor, dst_descriptor


def build_mapping_file(src_filename, dst_filename, mapping_filename,
                       method='bilinear', parallel_executable='mpirun',
                       mpi_tasks=1):
    """
    Call ``ESMF_RegridWeightGen`` to create a mapping file

    Parameters
    ----------
    src_filename : str
        The source lon/lat grid file

    dst_filename : str
        The destination MPAS mesh file

    mapping_filename : str
        The mapping file with interpolation weights to write out

    method : {'bilinear', 'neareststod', 'conserve'}, optional
        The method of interpolation used, see documentation for
        `ESMF_RegridWeightGen` for details.

    parallel_executable : {'srun', 'mpirun}, optional
        The name of the parallel executable to use.

    mpi_tasks : int, optional
        The number of MPI tasks
    """

    args = [parallel_executable, '-n', f'{mpi_tasks}', 'ESMF_RegridWeightGen',
            '--source', src_filename, '--destination', dst_filename,
            '--weight', mapping_filename, '--method', method, '--netcdf4',
            '--no_log', '--dst_loc', 'center', '--src_regional',
            '--dst_regional', '--ignore_unmapped']

    print('running: {}'.format(' '.join(args)))

    subprocess.check_call(args)


def remap_cosine_bell(src_filename, dst_filename, mapping_filename,
                      renormalize=None):
    """
    Call ``ncremap`` to remap data from the source grid to the destination mesh
    using the interpolation weights from the mapping file

    Parameters
    ----------
    src_filename : str
        The source data on lon/lat grid to be read

    dst_filename : str
        The destination data on an MPAS mesh to be written

    mapping_filename : str
        The mapping file with interpolation weights to be read

    renormalize : float, optional
        A threshold to use to renormalize the data
    """
    args = ['ncremap', '-m', mapping_filename,
            '-R', '--rgr col_nm=nCells']

    if renormalize is not None:
        args.append(f'--renormalize={renormalize}')

    args.extend([src_filename, dst_filename])

    print('running: {}'.format(' '.join(args)))

    subprocess.check_call(args)


def plot_area_weighted_sum(inputs, show=True):
    """
    Plot the area-weighted sum of the cosine bell function as a function
    of the total number of cells in the mesh

    Parameters
    ----------
    inputs : dict
        A nested dictionary with MPAS mesh names as outer keys , 'mesh' and
        'cosine_bell' as inner keys.  The corresponding values for the inner
        keys are an MPAS mesh file and the cosine bell field interpolated to
        the MPAS mesh, respectively.
    """

    area_weighted_sum = []
    ncells = []

    for mpas_mesh_name in inputs:
        mesh_filename = inputs[mpas_mesh_name]['mesh']
        mpas_filename = inputs[mpas_mesh_name]['cosine_bell']
        ds_mesh = xarray.open_dataset(mesh_filename)
        ncells.append(ds_mesh.sizes['nCells'])
        area_cell = ds_mesh.areaCell
        ds = xarray.open_dataset(mpas_filename)
        cosine_bell = ds.cosineBell

        area_weighted_sum.append((area_cell * cosine_bell).sum().values)

    if not show:
        plt.switch_backend('Agg')
    plt.plot(ncells, area_weighted_sum, '.-')
    plt.xlabel('number of mesh cells')
    plt.xlabel('area-weighted sum of cosine bell')
    if show:
        plt.show()
    else:
        plt.savefig('cosine_bell_sum.png')


def main():
    # Write out a file with the cosine bell function on a lon/lat grid
    lon_lat_filename = 'cosine_bell.nc'
    build_cosine_bell_map(lon_lat_filename, lon_center=0., radius=5e6)

    # Define some parameters for meshes to generate.  To add more meshes,
    # append additional dictionaries onto the list.  Or modify the existing
    # meshes to change the resolution at the North and/or South Pole, or to
    # run on more or fewer MPI tasks.
    meshes = [
        {'north_res': 1920,
         'south_res': 480,
         'mpi_tasks': 1},
        {'north_res': 960,
         'south_res': 240,
         'mpi_tasks': 2},
        {'north_res': 720,
         'south_res': 180,
         'mpi_tasks': 2},
        {'north_res': 480,
         'south_res': 120,
         'mpi_tasks': 4},
        {'north_res': 360,
         'south_res': 90,
         'mpi_tasks': 2},
        {'north_res': 300,
         'south_res': 75,
         'mpi_tasks': 4},
        {'north_res': 240,
         'south_res': 60,
         'mpi_tasks': 4}]

    # store the directory we start in so we can come back to it
    cwd = os.getcwd()

    # this is a nested dictionary with output files that we will use as inputs
    # to a later step
    outputs = dict()

    for mesh_info in meshes:
        north_res = mesh_info['north_res']
        south_res = mesh_info['south_res']
        mpi_tasks = mesh_info['mpi_tasks']

        # create a directory of the mesh (if it doesn't already exist) and
        # go into that directory
        base_directory = f'{north_res}_to_{south_res}'
        try:
            os.makedirs(base_directory)
        except FileExistsError:
            pass
        os.chdir(base_directory)

        directory = 'mesh'
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        os.chdir(directory)

        # build the MPAS mesh and store it in 'mesh.nc'
        mesh_filename = 'mesh.nc'
        mpas_mesh_name = f'MPAS{north_res}to{south_res}km'
        outputs[mpas_mesh_name] = dict()
        outputs[mpas_mesh_name]['mesh'] = \
            os.path.join(base_directory, directory, mesh_filename)
        build_mesh(north_res=float(north_res), south_res=float(south_res),
                   mesh_filename=mesh_filename)

        os.chdir('..')

        directory = 'mapping'
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        os.chdir(directory)

        # create two files that describe the lon/lat grid and MPAS mesh in
        # a different format called SCRIP
        symlink(os.path.join('..', 'mesh', mesh_filename), mesh_filename)
        symlink(os.path.join('..', '..', lon_lat_filename), lon_lat_filename)
        src_scrip_filename = 'src_scrip.nc'
        dst_scrip_filename = 'dst_scrip.nc'
        src_descriptor, dst_descriptor = write_scrip_files(
            lon_lat_filename=lon_lat_filename,
            mpas_mesh_filename=mesh_filename,
            mpas_mesh_name=mpas_mesh_name,
            src_scrip_filename=src_scrip_filename,
            dst_scrip_filename=dst_scrip_filename)
        lon_lat_grid_name = src_descriptor.meshName

        # in parallel, create a file containing interpolation weights that
        # can be used in the future to interpolate from the source lon/lat
        # grid to the destination MPAS mesh
        mapping_filename = \
            f'map_{lon_lat_grid_name}_to_{mpas_mesh_name}.nc'
        build_mapping_file(src_scrip_filename, dst_scrip_filename,
                           mapping_filename, method='bilinear',
                           parallel_executable='mpirun', mpi_tasks=mpi_tasks)

        os.chdir('..')

        directory = 'cosine_bell'
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
        os.chdir(directory)

        # interpolate the cosine_bell shape
        symlink(os.path.join('..', 'mesh', mesh_filename), mesh_filename)
        symlink(os.path.join('..', 'mapping', mapping_filename),
                mapping_filename)
        symlink(os.path.join('..', '..', lon_lat_filename), lon_lat_filename)
        mpas_filename = f'cosine_bell_{mpas_mesh_name}.nc'
        outputs[mpas_mesh_name]['cosine_bell'] = \
            os.path.join(base_directory, directory, mpas_filename)
        remap_cosine_bell(lon_lat_filename, mpas_filename, mapping_filename,
                          renormalize=0.01)

        extract_vtk(ignore_time=True,
                    variable_list=['cosineBell', 'areaCell', 'cellQuality'],
                    filename_pattern=mpas_filename,
                    mesh_filename=mesh_filename,
                    out_dir='vtk_cosine_bell')

        # go back to the original directory
        os.chdir(cwd)

    directory = 'area_weighted_sum'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    os.chdir(directory)

    # make symlinks to the contents of "outputs"
    inputs = dict()
    for mpas_mesh_name in outputs:
        inputs[mpas_mesh_name] = dict()

        mesh_filename = f'mesh_{mpas_mesh_name}.nc'
        inputs[mpas_mesh_name]['mesh'] = mesh_filename
        symlink(os.path.join('..', outputs[mpas_mesh_name]['mesh']),
                mesh_filename)

        mpas_filename = f'cosine_bell_{mpas_mesh_name}.nc'
        inputs[mpas_mesh_name]['cosine_bell'] = mpas_filename
        symlink(os.path.join('..', outputs[mpas_mesh_name]['cosine_bell']),
                mpas_filename)

    plot_area_weighted_sum(inputs)

    os.chdir(cwd)


if __name__ == '__main__':
    main()
