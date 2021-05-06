import os
import xarray


def file_complete(ds, fileName):
    """
    Find out if the file already has the same number of time slices as the
    monthly-mean data set
    """
    complete = False
    if os.path.exists(fileName):
        with xarray.open_dataset(fileName) as dsCompare:
            if ds.sizes['Time'] == dsCompare.sizes['Time']:
                complete = True

    return complete
