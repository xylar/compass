import os
import shutil
from importlib import resources

import netCDF4
import yaml

from compass.io import symlink
from compass.job import write_job_script
from compass.landice.extrapolate import extrapolate_variable
from compass.model import make_graph_file, run_model
from compass.step import Step


class EnsembleMember(Step):
    """
    A step for setting up a single ensemble member

    Attributes
    ----------
    run_num : integer
        the run number for this ensemble member

    name : str
        the name of the run being set up in this step

    ntasks : integer
        the number of parallel (MPI) tasks the step would ideally use

    test_resources_location : str
        path to the python package that contains the resources to be
        used for the test (namelist, streams, albany input file)

    input_file_name : str
        name of the input file that was read from the config

    basal_fric_exp : float
        value of basal friction exponent to use

    von_mises_threshold : float
        value of von Mises stress threshold to use

    calv_spd_lim : float
        value of calving speed limit to use

    gamma0 : float
        value of gamma0 to use in ISMIP6 ice-shelf basal melt param.

    deltaT : float
        value of deltaT to use in ISMIP6 ice-shelf basal melt param.
    """

    def __init__(self, test_case, run_num,
                 test_resources_location,
                 basal_fric_exp=None,
                 von_mises_threshold=None,
                 calv_spd_lim=None,
                 gamma0=None,
                 deltaT=None):
        """
        Creates a new run within an ensemble

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        run_num : integer
            the run number for this ensemble member

        test_resources_location : str
            path to the python package that contains the resources to be
            used for the test (namelist, streams, albany input file)

        basal_fric_exp : float
            value of basal friction exponent to use

        von_mises_threshold : float
            value of von Mises stress threshold to use

        calv_spd_lim : float
            value of calving speed limit to use

        gamma0 : float
            value of gamma0 to use in ISMIP6 ice-shelf basal melt param.

        deltaT : float
            value of deltaT to use in ISMIP6 ice-shelf basal melt param.
        """
        self.run_num = run_num
        self.test_resources_location = test_resources_location

        # store assigned param values for this run
        self.basal_fric_exp = basal_fric_exp
        self.von_mises_threshold = von_mises_threshold
        self.calv_spd_lim = calv_spd_lim
        self.gamma0 = gamma0
        self.deltaT = deltaT

        # define step (run) name
        self.name = f'run{run_num:03}'

        super().__init__(test_case=test_case, name=self.name)

    def setup(self):
        """
        Set up this run by setting up a baseline run configuration
        and then modifying parameters for this ensemble member based on
        an externally provided unit parameter vector
        """

        print(f'Setting up run number {self.run_num}')
        # print(f'    basal_fric_exp={self.basal_fric_exp}, '
        #       'von_mises_threshold={self.von_mises_threshold}, '
        #       'calv_spd_lim={self.calv_spd_lim}')

        # Get config for info needed for setting up simulation
        config = self.config
        section = config['ensemble']

        # save number of tasks to use
        # eventually compass could determine this, but for now we want
        # explicit control
        self.ntasks = section.getint('ntasks')

        # Set up base run configuration
        self.add_namelist_file(self.test_resources_location,
                               'namelist.landice')

        # copy over albany yaml file
        # cannot use add_input functionality because need to modify the file
        # in this function, and inputs don't get processed until after this
        # function
        with resources.path(self.test_resources_location,
                            'albany_input.yaml') as package_path:
            target = str(package_path)
            shutil.copy(target, self.work_dir)

        self.add_model_as_input()

        # modify param values as needed for this ensemble member

        # von Mises stress threshold
        options = {'config_grounded_von_Mises_threshold_stress':
                   f'{self.von_mises_threshold}',
                   'config_floating_von_Mises_threshold_stress':
                   f'{self.von_mises_threshold}'}

        # calving speed limit
        options['config_calving_speed_limit'] = \
            f'{self.calv_spd_lim}'

        # adjust basal friction exponent
        # rename and copy base file
        input_file_path = section.get('input_file_path')
        input_file_name = input_file_path.split('/')[-1]
        input_new_file_name = \
            f"{input_file_name.split('.')[:-1][0]}_MODIFIED_fricexp{self.basal_fric_exp:.4f}.nc"  # noqa E501
        self.input_file_name = input_new_file_name
        shutil.copy(input_file_path, os.path.join(self.work_dir,
                                                  input_new_file_name))
        # adjust mu and exponent
        orig_fric_exp = section.getfloat('orig_fric_exp')
        _adjust_friction_exponent(orig_fric_exp, self.basal_fric_exp,
                                  os.path.join(self.work_dir,
                                               input_new_file_name),
                                  os.path.join(self.work_dir,
                                               'albany_input.yaml'))
        # set input filename in streams and create streams file
        stream_replacements = {'input_file_init_cond': input_new_file_name}

        # adjust gamma0 and deltaT
        basal_melt_param_file_path = section.get('basal_melt_param_file_path')
        basal_melt_param_file_name = basal_melt_param_file_path.split('/')[-1]
        basal_melt_param_new_file_name = \
            f"{basal_melt_param_file_name.split('.')[:-1][0]}_MODIFIED_gamma{self.gamma0:.0f}_dT{self.deltaT:.3f}.nc"  # noqa E501
        shutil.copy(basal_melt_param_file_path,
                    os.path.join(self.work_dir,
                                 basal_melt_param_new_file_name))
        _adjust_basal_melt_params(os.path.join(self.work_dir,
                                  basal_melt_param_new_file_name),
                                  self.gamma0, self.deltaT)
        stream_replacements['basal_melt_param_file_name'] = \
            basal_melt_param_new_file_name

        # store modified namelist and streams options
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')
        self.add_streams_file(self.test_resources_location, 'streams.landice',
                              out_name='streams.landice',
                              template_replacements=stream_replacements)

        # set job name to run number so it will get set in batch script
        self.config.set('job', 'job_name', f'uq_{self.name}')
        machine = self.config.get('deploy', 'machine')
        write_job_script(self.config, machine,
                         target_cores=self.ntasks, min_cores=self.min_tasks,
                         work_dir=self.work_dir)

        # COMPASS does not create symlinks for the load script in step dirs,
        # so use the standard approach for creating that symlink in each
        # step dir.
        if 'LOAD_COMPASS_ENV' in os.environ:
            script_filename = os.environ['LOAD_COMPASS_ENV']
            # make a symlink to the script for loading the compass conda env.
            symlink(script_filename, os.path.join(self.work_dir,
                                                  'load_compass_env.sh'))

    def run(self):
        """
        Run this member of the ensemble.
        Eventually we want this function to handle restarts.
        """

        make_graph_file(mesh_filename=self.input_file_name,
                        graph_filename='graph.info')
        run_model(self)


def _adjust_friction_exponent(orig_fric_exp, new_fric_exp, filename,
                              albany_input_yaml):
    """
    Function to adjust the basal friction parameter by adjusting muFriction
    field in an input file to maintain the same basal shear stress, and
    then set the desired exponent value in the albany_input.yaml file.
    """
    f = netCDF4.Dataset(filename, 'r+')
    f.set_auto_mask(False)
    mu = f.variables['muFriction'][0, :]
    uX = f.variables['uReconstructX'][0, :, -1]
    uY = f.variables['uReconstructY'][0, :, -1]
    spd = (uX**2 + uY**2)**0.5 * (60. * 60. * 24. * 365.)
    mu = mu * spd**(orig_fric_exp - new_fric_exp)
    # The previous step leads to infs or nans in ice-free areas.
    # Set them all to 0.0 for the extrapolation step
    mu[spd == 0.0] = 0.0
    f.variables['muFriction'][0, :] = mu[:]
    f.close()

    extrapolate_variable(filename, 'muFriction', 'min')

    # now set exp in albany yaml file
    with open(albany_input_yaml, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # Change value
    (loaded['ANONYMOUS']['Problem']['LandIce BCs']['BC 0']
        ['Basal Friction Coefficient']
        ['Power Exponent']) = float(new_fric_exp)
    # write out again
    with open(albany_input_yaml, 'w') as stream:
        try:
            yaml.dump(loaded, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)


def _adjust_basal_melt_params(filename, gamma0=None, deltaT=None):
    """
    Function to adjust gamma0 and deltaT in a MALI input file for basal melt
    parameters.
    Currently assumes this is a regional mesh, and there is a single value of
    deltaT that should be used everywhere in the domain.
    """
    f = netCDF4.Dataset(filename, 'r+')
    f.set_auto_mask(False)
    if gamma0 is not None:
        f.variables['ismip6shelfMelt_gamma0'][:] = gamma0
    if deltaT is not None:
        f.variables['ismip6shelfMelt_deltaT'][:] = deltaT
    f.close()
