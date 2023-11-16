import xarray as xr

from compass.ocean.mesh.remap_topography import RemapTopography


class EC30to60SmoothTopo(RemapTopography):
    """
    A class for smoothing topography for EC30to60 meshes
    """
    def __init__(self, test_case, culled_mesh_step, mesh_name):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        culled_mesh_step : compass.ocean.mesh.cull.CullMeshStep
            The mesh step containing the mesh to remap to and the culled
            topography

        mesh_name : str, optional
            The name of the MPAS mesh to include in the mapping file
        """
        super().__init__(test_case=test_case, mesh_step=culled_mesh_step,
                         mesh_filename='culled_mesh.nc',
                         name='smooth_topography', mesh_name=mesh_name,
                         smooth=True)

        self.add_input_file(filename='previous_topography.nc',
                            target='../cull_mesh/topography_culled.nc')

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

        expand_const = self.config.getfloat('smooth_topography', 'expand_dist')
        depth_threshold = self.config.getfloat('smooth_topography',
                                               'depth_threshold')
        with xr.open_dataset('previous_topography.nc') as ds:
            # smooth by expand_const below depth_threshold bed elevation and
            # no smoothing above
            da_expand_dist = xr.where(ds.bed_elevation < depth_threshold,
                                      expand_const, 0.)

            ds_out = xr.Dataset()
            ds_out['expand_dist'] = da_expand_dist
            ds_out.write('expand_dist.nc')

            expand_dist = da_expand_dist.values

        return expand_dist
