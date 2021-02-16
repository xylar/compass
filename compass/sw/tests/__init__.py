from compass.sw.tests import case_1_cosine_bell_advection


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A dictionary of configurations within this core

    """
    testcases = list()
    for configuration in [case_1_cosine_bell_advection]:
        testcases.extend(configuration.collect())

    return testcases
