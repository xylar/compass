import os

import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.ocean.haney import compute_haney_number
from compass.ocean.iceshelf import compute_land_ice_pressure_and_draft
from compass.ocean.plot import plot_initial_state
from compass.ocean.tests.global_ocean.metadata import (
    add_mesh_and_init_metadata,
)
from compass.ocean.vertical import init_vertical_coord
from compass.ocean.vertical.fill import fill_zlevel_bathymetry_holes
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for baroclinic channel
    test cases

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.mesh.MeshStep
        The step for creating the mesh

    initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
        The initial condition dataset to use
    """
    def __init__(self, test_case, mesh, initial_condition):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
            The initial condition dataset to use
        """
        if initial_condition not in ['WOA23', 'PHC', 'EN4_1900']:
            raise ValueError(f'Unknown initial_condition {initial_condition}')

        super().__init__(test_case=test_case, name='initial_state')
        self.mesh = mesh
        self.initial_condition = initial_condition

        cull_step = self.mesh.steps['cull_mesh']
        target = os.path.join(cull_step.path, 'topography_culled.nc')
        self.add_input_file(filename='topography.nc',
                            work_dir_target=target)

        self.add_input_file(
            filename='wind_stress.nc',
            target='../remap_init/wind_stress_remapped.nc')

        for prefix in ['temperature', 'salinity']:
            self.add_input_file(
                filename=f'{prefix}_depth.nc',
                target=f'../remap_init/{prefix}_remapped.nc')

        mesh_path = self.mesh.get_cull_mesh_path()

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        if self.mesh.with_ice_shelf_cavities:
            self.add_input_file(
                filename='land_ice_mask.nc',
                work_dir_target=f'{mesh_path}/land_ice_mask.nc')

        self.add_model_as_input()

        for file in ['initial_state.nc', 'init_mode_forcing_data.nc',
                     'graph.info']:
            self.add_output_file(filename=file)

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        ds_mesh = xr.open_dataset('mesh.nc')

        ds = ds_mesh.copy()

        self._add_vertical_coordinate(ds)

        self._add_initial_state(ds)

        self._add_coriolis(ds)

        haney_edge, haney_cell = compute_haney_number(
            ds, ds.layerThickness, ds.ssh)
        ds['rx1Cell'] = haney_cell
        ds['rx1Edge'] = haney_edge

        write_netcdf(ds, 'initial_state.nc')

        self._compute_forcing(ds)

        add_mesh_and_init_metadata(self.outputs, config,
                                   init_filename='initial_state.nc')

        plot_initial_state(input_file_name='initial_state.nc',
                           output_file_name='initial_state.png')

    def _add_vertical_coordinate(self, ds):
        """ Add a vertical coordinate to a data set containing the mesh """

        config = self.config
        section = config['global_ocean_init']
        min_land_ice_fraction = section.getfloat('min_land_ice_fraction')
        min_column_thickness = section.getfloat('min_column_thickness')
        min_layer_thickness = section.getfloat('min_layer_thickness')
        min_levels = section.getint('minimum_levels')

        ds_topo = xr.open_dataset('topography.nc')
        bed_elevation = ds_topo.bed_elevation

        if self.mesh.with_ice_shelf_cavities:
            ssh = ds_topo.landIceDraftObserved
            ds['landIceFraction'] = \
                ds_topo.landIceFracObserved.expand_dims(dim='Time', axis=0)

            # This inequality needs to be > rather than >= to ensure
            # correctness when min_land_ice_fraction = 0
            mask = ds.landIceFraction > min_land_ice_fraction

            ds['landIceMask'] = mask.astype(int)
            ds['landIceFraction'] = xr.where(mask, ds.landIceFraction, 0.)

            ds['landIceFloatingMask'] = ds.landIceMask
            ds['landIceFloatingFraction'] = ds.landIceFraction

            ref_density = constants['SHR_CONST_RHOSW']
            land_ice_pressure, _ = compute_land_ice_pressure_and_draft(
                ssh=ssh, modify_mask=ssh < 0., ref_density=ref_density)

            ds['landIcePressure'] = land_ice_pressure
            ds['landIceDraft'] = ssh
        else:
            ssh = xr.zeros_like(ds_topo.bed_elevation)

        ds['ssh'] = ssh
        ds['bottomDepth'] = -bed_elevation

        # dig the bathymetry deeper where the column thickness is too shallow
        min_column_thickness = max(min_column_thickness,
                                   min_levels * min_layer_thickness)
        min_depth = -ssh + min_column_thickness
        ds['bottomDepth'] = np.maximum(ds.bottomDepth, min_depth)

        init_vertical_coord(config, ds)

        fill_zlevel_bathymetry_holes(ds)

        # this time, raise the ssh where the column thickness is too shallow
        # because we don't want to recreate the holes.  Note that this assumes
        # minLevelCell = 0 for now (i.e. z-star or Haney number vertical
        # coordinate).

        min_ssh = -ds.bottomDepth + min_column_thickness
        ds['ssh'] = np.maximum(ds.ssh, min_ssh)

    @staticmethod
    def _add_initial_state(ds):
        """ Add T and S to a data set containing mesh and ver. coord. """

        zmid = ds.zMid
        # interpolate T and S to zMid
        for var in ['temperature', 'salinity']:
            ds_var = xr.open_dataset(f'{var}_depth.nc')
            ds_var['nCells'] = ('nCells', np.arange(ds_var.sizes['nCells']))
            da = ds_var[var]
            # depths are positive, whereas zMid values are negative
            da = da.interp(depth=-zmid)
            ds[var] = da

        ds = ds.drop_vars(['nCells', 'depth'])

        normalVelocity = xr.zeros_like(ds.xEdge)
        normalVelocity = normalVelocity.broadcast_like(ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        ds['normalVelocity'] = normalVelocity.expand_dims(dim='Time', axis=0)

    @staticmethod
    def _add_coriolis(ds):
        """ Add T and S to a data set containing mesh and ver. coord. """

        sday = constants['SHR_CONST_SDAY']
        # SHR_CONST_OMEGA is not correct as of mpas_tools v0.20.0
        omega = 2. * np.pi / sday

        for geom in ['Cell', 'Edge', 'Vertex']:
            ds[f'f{geom}'] = 2. * omega * np.sin(ds[f'lat{geom}'])

    def _compute_forcing(self, ds):
        config = self.config
        section = config['global_ocean_init']
        piston_velocity = section.getfloat('piston_velocity')
        interior_restore_rate = section.getfloat('interior_restore_rate')
        ds_forcing = xr.open_dataset('wind_stress.nc')
        ds_forcing['temperatureSurfaceRestoringValue'] = \
            ds['temperature'].isel(nVertLevels=0)
        ds_forcing['salinitySurfaceRestoringValue'] = \
            ds['salinity'].isel(nVertLevels=0)
        ds_forcing['temperaturePistonVelocity'] = \
            piston_velocity * xr.ones_like(
                ds_forcing.temperatureSurfaceRestoringValue)
        ds_forcing['salinityPistonVelocity'] = \
            ds_forcing.temperaturePistonVelocity
        ds_forcing['temperatureInteriorRestoringRate'] = \
            interior_restore_rate * xr.ones_like(ds.zMid)
        ds_forcing['salinityInteriorRestoringRate'] = \
            ds_forcing.temperatureInteriorRestoringRate
        ds_forcing['temperatureInteriorRestoringValue'] = ds['temperature']
        ds_forcing['salinityInteriorRestoringValue'] = ds['salinity']

        write_netcdf(ds_forcing, 'init_mode_forcing_data.nc')
