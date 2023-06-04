import xarray as xr
from mpas_tools.io import write_netcdf
from pyremap import LatLonGridDescriptor, MpasMeshDescriptor, Remapper

from compass.step import Step


class RemapInit(Step):
    """
    A step for remapping an initial condition and forcing dataset from a
    latitude-longitude grid to a global MPAS-Ocean mesh

    Attributes
    ----------
    cull_mesh_step : compass.ocean.mesh.cull.CullMeshStep
        The cull mesh step containing input files to this step

    mesh_name : str
        The name of the MPAS mesh to include in the mapping file

    initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
        The initial condition dataset to use

    remap : dict
        A nested dictionary of variables to remap
    """

    def __init__(self, test_case, cull_mesh_step, initial_condition,
                 mesh_name, name='remap_init', subdir=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.init.Init
            The test case this step belongs to

        cull_mesh_step : compass.ocean.mesh.cull.CullMeshStep
            The base mesh step containing input files to this step

        initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
            The initial condition dataset to use

        mesh_name : str
            The name of the MPAS mesh to include in the mapping file

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        """
        super().__init__(test_case, name=name, subdir=subdir,
                         ntasks=None, min_tasks=None)
        self.cull_mesh_step = cull_mesh_step
        self.initial_condition = initial_condition
        self.mesh_name = mesh_name

        base_path = self.cull_mesh_step.path
        target = f'{base_path}/culled_mesh.nc'
        self.add_input_file(filename='mesh.nc', work_dir_target=target)

        self.add_input_file(
            filename='wind_stress.nc',
            target='windStress.ncep_1958-2000avg.interp3600x2431.151106.nc',
            database='initial_condition_database')

        remap = dict(
            wind_stress=dict(
                var_list=['TAUX', 'TAUY'],
                mpas_vars=['windStressZonal', 'windStressMeridional'],
                lon='u_lon',
                lat='u_lat'))

        if initial_condition == 'WOA23':
            self.add_input_file(
                filename='temperature.nc',
                target='woa23_decav_0.25_extrap.20230416.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='woa23_decav_0.25_extrap.20230416.nc',
                database='initial_condition_database')
            remap['temperature'] = dict(
                var_list=['pt_an'],
                mpas_vars=['temperature'],
                lon='lon',
                lat='lat',
                depth='depth')
            remap['salinity'] = dict(
                var_list=['s_an'],
                mpas_vars=['salinity'],
                lon='lon',
                lat='lat',
                depth='depth')
        elif initial_condition == 'PHC':
            self.add_input_file(
                filename='temperature.nc',
                target='PotentialTemperature.01.filled.60levels.PHC.151106.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='Salinity.01.filled.60levels.PHC.151106.nc',
                database='initial_condition_database')
            remap['temperature'] = dict(
                var_list=['TEMP'],
                mpas_vars=['temperature'],
                lon='t_lon',
                lat='t_lat',
                depth='depth_t')
            remap['salinity'] = dict(
                var_list=['SALT'],
                mpas_vars=['salinity'],
                lon='t_lon',
                lat='t_lat',
                depth='depth_t')
        else:
            # EN4_1900
            self.add_input_file(
                filename='temperature.nc',
                target='PotentialTemperature.100levels.Levitus.'
                       'EN4_1900estimate.200813.nc',
                database='initial_condition_database')
            self.add_input_file(
                filename='salinity.nc',
                target='Salinity.100levels.Levitus.EN4_1900estimate.200813.nc',
                database='initial_condition_database')
            remap['temperature'] = dict(
                var_list=['TEMP'],
                mpas_vars=['temperature'],
                lon='t_lon',
                lat='t_lat',
                depth='depth_t')
            remap['salinity'] = dict(
                var_list=['SALT'],
                mpas_vars=['salinity'],
                lon='t_lon',
                lat='t_lat',
                depth='depth_t')

        self.remap = remap

        self.add_output_file(filename='temperature_remapped.nc')
        self.add_output_file(filename='salinity_remapped.nc')
        self.add_output_file(filename='wind_stress_remapped.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()
        config = self.config
        self.ntasks = config.getint('global_ocean_init', 'remap_ntasks')
        self.min_tasks = config.getint('global_ocean_init', 'remap_min_tasks')

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
        self.ntasks = config.getint('global_ocean_init', 'remap_ntasks')
        self.min_tasks = config.getint('global_ocean_init', 'remap_min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        parallel_executable = config.get('parallel', 'parallel_executable')
        remap = self.remap

        out_mesh_name = self.mesh_name
        out_descriptor = MpasMeshDescriptor(fileName='mesh.nc',
                                            meshName=self.mesh_name)

        method = 'bilinear'

        for file_prefix in remap:
            remap_info = remap[file_prefix]
            lon = remap_info['lon']
            lat = remap_info['lat']
            in_filename = f'{file_prefix}.nc'
            ncremap_filename = f'{file_prefix}_ncremap.nc'
            out_filename = f'{file_prefix}_remapped.nc'

            in_descriptor = LatLonGridDescriptor.read(
                fileName=in_filename,
                lonVarName=lon,
                latVarName=lat)

            in_mesh_name = in_descriptor.meshName

            mapping_file_name = \
                f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
            remapper = Remapper(in_descriptor, out_descriptor,
                                mapping_file_name)

            remapper.build_mapping_file(method=method, mpiTasks=self.ntasks,
                                        tempdir='.', logger=logger,
                                        esmf_parallel_exec=parallel_executable)

            remapper.remap_file(inFileName=in_filename,
                                outFileName=ncremap_filename,
                                variableList=remap_info['var_list'],
                                logger=logger)

            ds_in = xr.open_dataset(ncremap_filename)
            # we don't want to keep lon and lat as coordinates
            ds_in = ds_in.reset_coords([lon, lat], drop=True)
            ds_in = ds_in.rename({'ncol': 'nCells'})
            if 'depth' in remap_info and remap_info['depth'] != 'depth':
                ds_in = ds_in.rename({remap_info['depth']: 'depth'})
            ds_out = xr.Dataset()
            for index, var in enumerate(remap_info['var_list']):
                mpas_var = remap_info['mpas_vars'][index]
                ds_out[mpas_var] = ds_in[var]

            write_netcdf(ds_out, out_filename)
