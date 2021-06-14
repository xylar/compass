from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of the Greenland Ice Sheet setup,
    one with one core and one with eight.  The test case verifies that the
    results of the two runs are identical.
    """

    def __init__(self, test_group, velo_solver):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.greenland.Greenland
            The test group that this test case belongs to

        velo_solver : {'sia', 'FO'}
            The velocity solver to use for the test case
        """
        name = 'decomposition_test'
        subdir = '{}_{}'.format(velo_solver.lower(), name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        if velo_solver == 'sia':
            self.cores_set = [1, 8]
        elif velo_solver == 'FO':
            self.cores_set = [16, 32]

        for procs in self.cores_set:
            name = '{}proc_run'.format(procs)
            self.add_step(
                RunModel(test_case=self, velo_solver=velo_solver, name=name,
                         subdir=name, cores=procs, min_cores=procs,
                         threads=1))

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['thickness', 'normalVelocity']
        steps = self.steps_to_run
        name1 = '{}proc_run'.format(self.cores_set[0])
        name2 = '{}proc_run'.format(self.cores_set[1])
        if name1 in steps and name2 in steps:
            compare_variables(test_case=self, variables=variables,
                              filename1='{}/output.nc'.format(name1),
                              filename2='{}/output.nc'.format(name2))
        else:
            assert False, "Error in decomposition test directory structure"
