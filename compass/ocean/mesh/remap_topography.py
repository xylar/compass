import os

import xarray as xr
from mpas_tools.io import write_netcdf
from pyremap import LatLonGridDescriptor, MpasCellMeshDescriptor, Remapper

from compass.step import Step


class RemapTopography(Step):
    """
    A step for remapping bathymetry and ice-shelf topography from a
    latitude-longitude grid to a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh_step : compass.Step
        The mesh step containing the mesh to remap to

    mesh_filename : str
        The filename within ``mesh_step`` that contains the mesh

    mesh_name : str
        The name of the MPAS mesh to include in the mapping file

    smooth : bool
        Whether to smooth the topography
    """

    def __init__(self, test_case, mesh_step, mesh_filename='base_mesh.nc',
                 name='remap_topography', subdir=None, mesh_name='MPAS_mesh',
                 smooth=False):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        mesh_step : compass.Step
            The mesh step containing the mesh to remap to

        mesh_filename : str, optional
            The filename within ``mesh_step`` that contains the mesh

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        mesh_name : str, optional
            The name of the MPAS mesh to include in the mapping file

        smooth : bool, optional
            Whether to smooth the topography
        """
        super().__init__(test_case, name=name, subdir=subdir,
                         ntasks=None, min_tasks=None)
        self.mesh_step = mesh_step
        self.mesh_filename = mesh_filename
        self.mesh_name = mesh_name
        self.smooth = smooth

        self.add_output_file(filename='topography_remapped.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()
        topo_filename = self.config.get('remap_topography', 'topo_filename')
        self.add_input_file(
            filename='topography.nc',
            target=topo_filename,
            database='bathymetry_database')

        target = os.path.join(self.mesh_step.path, self.mesh_filename)
        self.add_input_file(filename='mesh.nc', work_dir_target=target)

        config = self.config
        self.ntasks = config.getint('remap_topography', 'ntasks')
        self.min_tasks = config.getint('remap_topography', 'min_tasks')

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        config = self.config
        self.ntasks = config.getint('remap_topography', 'ntasks')
        self.min_tasks = config.getint('remap_topography', 'min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        parallel_executable = config.get('parallel', 'parallel_executable')

        lon_var = config.get('remap_topography', 'lon_var')
        lat_var = config.get('remap_topography', 'lat_var')
        method = config.get('remap_topography', 'method')
        renorm_threshold = config.getfloat('remap_topography',
                                           'renorm_threshold')

        in_descriptor = LatLonGridDescriptor.read(fileName='topography.nc',
                                                  lonVarName=lon_var,
                                                  latVarName=lat_var)

        in_mesh_name = in_descriptor.meshName

        out_mesh_name = self.mesh_name
        out_descriptor = MpasCellMeshDescriptor(fileName='mesh.nc',
                                                meshName=self.mesh_name)

        mapping_file_name = \
            f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
        remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

        if self.smooth:
            expand_dist = self.build_expand_dist()
            expand_factor = self.build_expand_factor()
        else:
            expand_dist = None
            expand_factor = None

        remapper.build_mapping_file(method=method, mpiTasks=self.ntasks,
                                    tempdir='.', logger=logger,
                                    esmf_parallel_exec=parallel_executable,
                                    expandDist=expand_dist,
                                    expandFactor=expand_factor)

        remapper.remap_file(inFileName='topography.nc',
                            outFileName='topography_ncremap.nc',
                            logger=logger)

        ds_in = xr.open_dataset('topography_ncremap.nc')
        ds_in = ds_in.rename({'ncol': 'nCells'})
        ds_out = xr.Dataset()
        rename = {'bathymetry_var': 'bed_elevation',
                  'ice_draft_var': 'landIceDraftObserved',
                  'ice_thickness_var': 'landIceThkObserved',
                  'ice_frac_var': 'landIceFracObserved',
                  'grounded_ice_frac_var': 'landIceGroundedFracObserved',
                  'ocean_frac_var': 'oceanFracObserved'}

        for option in rename:
            in_var = config.get('remap_topography', option)
            out_var = rename[option]
            ds_out[out_var] = ds_in[in_var]

        # renormalize elevation variables
        norm = ds_out.oceanFracObserved
        valid = norm > renorm_threshold
        for var in ['bed_elevation', 'landIceDraftObserved',
                    'landIceThkObserved']:
            ds_out[var] = xr.where(valid, ds_out[var] / norm, 0.)

        write_netcdf(ds_out, 'topography_remapped.nc')

    def build_expand_dist(self):
        """
        Get the distance in meters over which to expand MPAS cells if smoothing
        is performed.  The default behavior is to return the value of the
        ``expand_dist`` config option but this method can be overridden to
        provide a value for each cell.

        Returns
        -------
        expand_dist : float or numpy.ndarray
            the distance over which to expand MPAS cells
        """

        expand_dist = self.config.getfloat('smooth_topography', 'expand_dist')
        return expand_dist

    def build_expand_factor(self):
        """
        Get the factor by which to expand MPAS cells if smoothing is
        performed.  The default behavior is to return the value of the
        ``expand_factor`` config option but this method can be overridden to
        provide a value for each cell.

        Returns
        -------
        expand_factor : float or numpy.ndarray
            the factor by which to expand MPAS cells
        """

        expand_factor = self.config.getfloat('smooth_topography',
                                             'expand_factor')
        return expand_factor
