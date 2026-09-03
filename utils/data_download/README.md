Using `download_gdex`
=====================

`download_gdex` is an executable module that constructs API requests for the
[NCAR Geoscience Data Exchange (GDEX)](https://gdex.ucar.edu/) data repository,
where for example, NCEP gridded forecasts and reanalysis are archived.

This module currently only defines requests for the NCEP Climate Forecast System
Reanalysis (CFSR) and Climate Forecast System Version 2 (CFSv2). Additional
definitions should be added to `gdex_codes.yaml`.

A GDEX token is required for data downloads. A token can be retrieved at
https://gdex.ucar.edu/accounts/profile/ (account required).

Instructions
------------

   1. Navigate to the directory where data will be downloaded, and symlink
      `download_gdex.py` and `gdex_codes.yaml`.
      ```
      cd /path/to/download
      ln -s /path/to/download_gdex.py ./
      ln -s /path/to/gdex_codes.yaml ./
      ```
    
   2. Initiate the request by executing `download_gdex.py`. The options are:
      ```
      python download_gdex.py --help

      usage: download_gdex.py [-h] [-f FREQUENCY] [-b REGION_BOX] parameter_group model start_date end_date

      Download CFS data from gdex.ucar.edu

      positional arguments:
      parameter_group       Name of parameter group (e.g., wind)
      model                 Model name (CFSR or CFSv2)
      start_date            Start date (yyyy-mm-dd)
      end_date              End date (yyyy-mm-dd)

      options:
      -h, --help            show this help message and exit
      -f, --frequency FREQUENCY
                            Frequency (e.g., hourly)
      -b, --region_box REGION_BOX
                            Bounding box [wlon, elon, slat, nlat]
      ```
      Options for `parameter_group` can be browsed in `gdex_codes.yaml`. To
      download u and v wind velocity from the CFSv2 model from 2011-01-01 to
      2014-01-01, for example, the command would be the following
      ```
      python download_gdex.py wind CFSv2 2011-01-01 2014-01-01
      ```