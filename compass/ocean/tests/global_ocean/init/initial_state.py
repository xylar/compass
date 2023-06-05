import os

import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.ocean.iceshelf import compute_land_ice_pressure_and_draft
# from compass.ocean.plot import plot_initial_state, plot_vertical_grid
# from compass.ocean.tests.global_ocean.metadata import (
#     add_mesh_and_init_metadata,
# )
from compass.ocean.vertical import init_vertical_coord
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

        for prefix in ['wind_stress', 'temperature', 'salinity']:
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
        section = config['global_ocean_init']
        min_land_ice_fraction = section.getfloat('min_land_ice_fraction')
        min_column_thickness = section.getfloat('min_column_thickness')
        min_layer_thickness = section.getfloat('min_layer_thickness')
        min_levels = section.getint('minimum_levels')

        ds_mesh = xr.open_dataset('mesh.nc')

        ds_topo = xr.open_dataset('topography.nc')
        ssh = ds_topo.landIceDraftObserved
        bed_elevation = ds_topo.bed_elevation

        ds = ds_mesh.copy()

        ds['landIceFraction'] = \
            ds_topo.landIceFracObserved.expand_dims(dim='Time', axis=0)
        ds['landIceFloatingFraction'] = ds['landIceFraction']

        # This inequality needs to be > rather than >= to ensure correctness
        # when min_land_ice_fraction = 0
        mask = ds.landIceFraction > min_land_ice_fraction

        floating_mask = np.logical_and(
            ds.landIceFloatingFraction > 0,
            ds.landIceFraction > min_land_ice_fraction)

        ds['landIceMask'] = mask.astype(int)
        ds['landIceFloatingMask'] = floating_mask.astype(int)

        ds['landIceFraction'] = xr.where(mask, ds.landIceFraction, 0.)

        ref_density = constants['SHR_CONST_RHOSW']
        land_ice_pressure, _ = compute_land_ice_pressure_and_draft(
            ssh=ssh, modify_mask=ssh < 0., ref_density=ref_density)

        ds['landIcePressure'] = land_ice_pressure
        ds['landIceDraft'] = ssh
        ds['ssh'] = ssh
        ds['bottomDepth'] = -bed_elevation

        min_column_thickness = max(min_column_thickness,
                                   min_levels * min_layer_thickness)
        min_depth = -ssh + min_column_thickness
        ds['bottomDepth'] = np.maximum(ds.bottomDepth, min_depth)

        init_vertical_coord(config, ds)

        write_netcdf(ds, 'initial_state.nc')

        # for prefix in ['temperature', 'salinity']:

        # add_mesh_and_init_metadata(self.outputs, config,
        #                            init_filename='initial_state.nc')

        # plot_initial_state(input_file_name='initial_state.nc',
        #                    output_file_name='initial_state.png')
