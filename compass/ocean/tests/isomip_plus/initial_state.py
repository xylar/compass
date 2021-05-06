import xarray
import numpy
import cmocean

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.translate import translate
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.cime.constants import constants

from compass.step import Step
from compass.ocean.vertical import generate_grid
from compass.ocean.iceshelf import compute_land_ice_pressure_and_draft
from compass.ocean.vertical.zstar import compute_layer_thickness_and_zmid
from compass.ocean.tests.isomip_plus.geom import process_input_geometry, \
    interpolate_geom, interpolate_ocean_mask
from compass.ocean.tests.isomip_plus.viz.plot import MoviePlotter


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for ISOMIP + test cases

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment

    vertical_coordinate : {'z-star', 'z-level', 'haney-number'}
        The vertical coordinate
    """
    def __init__(self, test_case, resolution, experiment, vertical_coordinate):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment

        vertical_coordinate : {'z-star', 'z-level', 'haney-number'}
            The vertical coordinate
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution
        self.experiment = experiment
        self.vertical_coordinate = vertical_coordinate

        if experiment in ['Ocean0', 'Ocean1']:
            self.add_input_file(filename='input_geometry.nc',
                                target='Ocean1_input_geom_v1.01.nc',
                                database='initial_condition_database')
        elif experiment == 'Ocean2':
            self.add_input_file(filename='input_geometry.nc',
                                target='Ocean2_input_geom_v1.01.nc',
                                database='initial_condition_database')
        else:
            raise ValueError('Unknown ISOMIP+ experiment {}'.format(
                experiment))

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc', 'init_mode_forcing_data.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['isomip_plus']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')
        filter_sigma = section.getfloat('topo_smoothing')*self.resolution
        min_ice_thickness = section.getfloat('min_ice_thickness')
        draft_scaling = section.getfloat('draft_scaling')

        process_input_geometry('input_geometry.nc',
                               'input_geometry_processed.nc',
                               filterSigma=filter_sigma,
                               minIceThickness=min_ice_thickness,
                               scale=draft_scaling)

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=False)
        translate(mesh=dsMesh, yOffset=-2*dc)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsGeom = xarray.open_dataset('input_geometry_processed.nc')

        min_ocean_fraction = config.getfloat('isomip_plus',
                                             'min_ocean_fraction')

        dsMask = interpolate_ocean_mask(dsMesh, dsGeom, min_ocean_fraction)
        dsMesh = cull(dsMesh, dsInverse=dsMask, logger=logger)
        dsMesh.attrs['is_periodic'] = 'NO'

        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        ds = interpolate_geom(dsMesh, dsGeom, min_ocean_fraction)

        if self.vertical_coordinate == 'z-star':
            self._compute_z_star(ds)
        else:
            raise ValueError('Vertical coordinate {} not supported '
                             '(yet).'.format(self.vertical_coordinate))

        for var in ['landIceFraction']:
            ds[var] = ds[var].expand_dims(dim='Time', axis=0)

        ref_density = constants['SHR_CONST_RHOSW']
        landIcePressure, landIceDraft = compute_land_ice_pressure_and_draft(
            ssh=ds.ssh, modify_mask=ds.ssh < 0., ref_density=ref_density)

        ds['landIcePressure'] = landIcePressure
        ds['landIceDraft'] = landIceDraft

        max_bottom_depth = -config.getfloat('vertical_grid', 'bottom_depth')
        frac = (0. - ds.zMid) / (0. - max_bottom_depth)

        # compute T, S
        init_top_temp = section.getfloat('init_top_temp')
        init_bot_temp = section.getfloat('init_bot_temp')
        init_top_sal = section.getfloat('init_top_sal')
        init_bot_sal = section.getfloat('init_bot_sal')
        ds['temperature'] = (1.0 - frac) * init_top_temp + frac * init_bot_temp
        ds['salinity'] = (1.0 - frac) * init_top_sal + frac * init_bot_sal

        # compute coriolis
        coriolis_parameter = section.getfloat('coriolis_parameter')

        ds['fCell'] = coriolis_parameter*xarray.ones_like(ds.xCell)
        ds['fEdge'] = coriolis_parameter*xarray.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter*xarray.ones_like(ds.xVertex)

        write_netcdf(ds, 'initial_state.nc')

        # plot a few fields
        plotter = MoviePlotter(inFolder=self.work_dir,
                               outFolder='{}/plots'.format(self.work_dir),
                               dsMesh=ds, ds=ds)

        plotter.plot_3d_field_top_bot_section(
            ds.zMid, nameInTitle='zMid', prefix='zmid', units='m',
            vmin=-720., vmax=0., cmap='cmo.deep_r')

        plotter.plot_3d_field_top_bot_section(
            ds.temperature, nameInTitle='temperature', prefix='temp',
            units='C', vmin=-2., vmax=1., cmap='cmo.thermal')

        plotter.plot_3d_field_top_bot_section(
            ds.salinity, nameInTitle='salinity', prefix='salin',
            units='PSU', vmin=33.8, vmax=34.7, cmap='cmo.haline')

        # compute restoring
        dsForcing = xarray.Dataset()

        restore_top_temp = section.getfloat('restore_top_temp')
        restore_bot_temp = section.getfloat('restore_bot_temp')
        restore_top_sal = section.getfloat('restore_top_sal')
        restore_bot_sal = section.getfloat('restore_bot_sal')
        dsForcing['temperatureInteriorRestoringValue'] = \
            (1.0 - frac) * restore_top_temp + frac * restore_bot_temp
        dsForcing['salinityInteriorRestoringValue'] = \
            (1.0 - frac) * restore_top_sal + frac * restore_bot_sal

        restore_rate = section.getfloat('restore_rate')
        restore_xmin = section.getfloat('restore_xmin')
        restore_xmax = section.getfloat('restore_xmax')
        frac = numpy.maximum(
            (ds.xCell - restore_xmin)/(restore_xmax-restore_xmin), 0.)

        # convert from 1/days to 1/s
        dsForcing['temperatureInteriorRestoringRate'] = \
            frac * restore_rate / constants['SHR_CONST_CDAY']
        dsForcing['salinityInteriorRestoringRate'] = \
            dsForcing['temperatureInteriorRestoringRate']

        # compute "evaporation"
        restore_evap_rate = section.getfloat('restore_evap_rate')

        mask = numpy.logical_and(ds.xCell >= restore_xmin,
                                 ds.xCell <= restore_xmax)
        # convert to m/s, negative for evaporation rather than precipitation
        evap_rate = -restore_evap_rate/(constants['SHR_CONST_CDAY']*365)
        # PSU*m/s to kg/m^2/s
        sflux_factor = 1.
        # C*m/s to W/m^2
        hflux_factor = 1./(ref_density*constants['SHR_CONST_CPSW'])
        dsForcing['evaporationFlux'] = mask*ref_density*evap_rate
        dsForcing['seaIceSalinityFlux'] = \
            mask*evap_rate*restore_top_sal/sflux_factor
        dsForcing['seaIceHeatFlux'] = \
            mask*evap_rate*restore_top_temp/hflux_factor

        write_netcdf(dsForcing, 'init_mode_forcing_data.nc')

    def _compute_z_star(self, ds):
        """Initialize with a z-star vertical coordinate"""
        config = self.config

        section = config['isomip_plus']

        min_column_thickness = section.getfloat('min_column_thickness')
        min_levels = section.getint('minimum_levels')

        interfaces = generate_grid(config=config)
        bottom_depth = interfaces[-1]

        ds['refBottomDepth'] = ('nVertLevels', interfaces[1:])
        ds['refTopDepth'] = ('nVertLevels', interfaces[0:-1])
        ds['refZMid'] = ('nVertLevels',
                         -0.5 * (interfaces[1:] + interfaces[0:-1]))
        ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

        # MPAS-Ocean expects bottom depth to be positive
        ds['bottomDepthObserved'] = -ds.bottomDepthObserved

        # Deepen the bottom depth to maintain the minimum water-column
        # thickness
        min_depth = numpy.maximum(-ds.ssh + min_column_thickness,
                                  ds.refBottomDepth[min_levels])
        ds['bottomDepth'] = numpy.maximum(ds.bottomDepthObserved, min_depth)

        # If there is anywhere that the bottomDepth has been forced below
        # max_bottom_depth, we need to adjust both the bottom depth and ssh
        ds['bottomDepth'] = numpy.minimum(ds.bottomDepth, bottom_depth)
        ds['ssh'] = numpy.maximum(ds.ssh,
                                  -bottom_depth + min_column_thickness)

        ds['cellMask'] = (ds.refTopDepth < ds.bottomDepth).transpose(
            'nCells', 'nVertLevels')

        ds['maxLevelCell'] = ds.cellMask.sum(dim='nVertLevels')
        if not numpy.all(ds.maxLevelCell >= 1):
            raise ValueError(
                'Something went wrong with culling.  There are still '
                'non-ocean cells in the culled mesh.')

        restingThickness, layerThickness, zMid = \
            compute_layer_thickness_and_zmid(
                ds.cellMask, ds.refBottomDepth, ds.bottomDepth,
                ds.maxLevelCell-1, ssh=ds.ssh)

        ds['layerThickness'] = layerThickness.expand_dims(dim='Time', axis=0)
        ds['ssh'] = ds.ssh.expand_dims(dim='Time', axis=0)
        ds['zMid'] = zMid.expand_dims(dim='Time', axis=0)
        ds['restingThickness'] = restingThickness
