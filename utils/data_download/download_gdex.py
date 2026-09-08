import argparse
import os
import time

import requests
import yaml
from dateutil.parser import parse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Download CFS data from gdex.ucar.edu')
    parser.add_argument(
        'parameter_group', help='Name of parameter group (e.g., wind)')
    parser.add_argument('model', help='Model name (CFSR or CFSv2)')
    parser.add_argument('start_date', help='Start date (yyyy-mm-dd)')
    parser.add_argument('end_date', help='End date (yyyy-mm-dd)')
    parser.add_argument(
        '-f', '--frequency', default='hourly', help='Frequency (e.g., hourly)')
    parser.add_argument(
        '-b', '--region_box', default=None,
        help='Bounding box [wlon, elon, slat, nlat]')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    download_cfs_data(
        args.parameter_group,
        args.model,
        args.start_date,
        args.end_date,
        frequency=args.frequency,
        region_box=args.region_box,
    )


def download_cfs_data(parameter_group, model, start_date, end_date,
                      frequency='hourly', region_box=None):
    """
    Download CFS data from gdex.ucar.edu

    Attributes
    ----------
    parameter_group : str
        Label for the desired parameter group (e.g., 'wind')

    model : str
        Label for the desired model ('CFSR' or 'CFSv2')

    start_date : str
        Start date

    end_date : str
        End date

    frequency : str
        Desired frequency (default 'hourly')

    region_box : list
        Bounding box [wlon, elon, slat, nlat] (default global)
    """

    if region_box is None:
        region_box = ['-180', '180', '-90', '90']

    fmt = '%Y%m%d%H%M'
    date_range = [parse(dt).strftime(fmt) for dt in (start_date, end_date)]

    # Load GDEX codes definitions
    with open('gdex_codes.yaml', 'r') as f:
        codes = yaml.safe_load(f)
    dataset = codes['datasets'][model][frequency]
    parameter_codes = codes['parameter_groups'][parameter_group]

    # Build grid codes
    grid = parameter_codes['grid']
    if grid == 'full':
        if model == 'CFSR':
            grid = 'T382'
        elif model == 'CFSv2':
            grid = 'T574'
    grid_codes = codes['grids'][grid]

    # Build product code list (Hours 1-6)
    product_type = parameter_codes['product']
    products = []
    for hour in range(1, 7):
        if product_type == '1-hour Average':
            product = f'{product_type} (initial+{hour - 1} to initial+{hour})'
        else:
            product = f'{hour}-hour {product_type}'
            if product_type != 'Forecast':
                product = f'{product} (initial+0 to initial+1)'
        if 'pressure' in parameter_group and hour == 6:
            products.append('Analysis')
        else:
            products.append(product)

    # Build request dict
    control = {
        'dataset': dataset,
        'date': '/to/'.join(date_range),
        'param': '/'.join(parameter_codes['parameters']),
        'level': parameter_codes['level'],
        'product': '/'.join(products),
        'oformat': 'netCDF',
        'nlat': region_box[3],
        'slat': region_box[2],
        'wlon': region_box[0],
        'elon': region_box[1],
        'gridproj': grid_codes[0],
        'griddef': grid_codes[1],
    }

    submit_request(control)


def submit_request(control):
    """
    Submit download request to GDEX API

    Attributes
    ----------
    control : dict
        Dictionary of request parameters
    """

    token = get_token()

    print('Submitting request to GDEX...')
    print(f'  Dataset  : {control['dataset']}')
    print(f'  Date     : {control['date']}')
    print(f'  Params   : {control['param']}')
    print(f'  Level    : {control['level']}')
    print(f'  Product  : {control['product']}')

    ret = api_post('submit/', token, control)
    if ret.status_code != 200:
        print(f'Submit failed (HTTP {ret.status_code}):\n{ret.text}')
        raise SystemExit(1)

    result = ret.json()
    if result.get('status') != 'ok':
        print(f'Submit error:\n{result}')
        raise SystemExit(1)

    request_idx = result['data']['request_id']
    print(f'Request accepted. ID: {request_idx}')

    print('Waiting for data preparation (this may take several minutes)...')
    while True:
        st = api_get(f'status/{request_idx}/', token).json()
        state = st['data']['status']
        print(f'  Status: {state}', flush=True)
        if state == 'Completed':
            break
        if state in ('Error', 'Cancelled'):
            print(f'Request ended with status: {state}')
            raise SystemExit(1)
        time.sleep(60)

    print('Fetching file list...')
    fl = api_get(f'get_req_files/{request_idx}/', token).json()
    web_files = [f['web_path'] for f in fl['data']['web_files']]
    print(f'{len(web_files)} file(s) to download.')

    for url in web_files:
        filename = os.path.basename(url)
        print(f'Downloading {filename} ...', end=' ', flush=True)
        resp = requests.get(url, stream=True)
        with open(filename, 'wb') as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        print('done')

    print('All files downloaded.')


def get_token():
    """Return cached API token, or prompt the user to paste one.

    Get your token from: https://gdex.ucar.edu/accounts/profile/
    """
    TOKEN_FILE = './gdex_token.txt'
    if os.path.isfile(TOKEN_FILE) and os.path.getsize(TOKEN_FILE) > 0:
        with open(TOKEN_FILE) as fh:
            return fh.read().strip()
    print('No GDEX token found.')
    print('Please log in at https://gdex.ucar.edu and visit:')
    print('  https://gdex.ucar.edu/accounts/profile/')
    print('then copy your API token.')
    token = input('Paste token here: ').strip()
    with open(TOKEN_FILE, 'w') as fh:
        fh.write(token)
    return token


def api_get(endpoint, token):
    BASE_URL = 'https://gdex.ucar.edu/api/'
    url = BASE_URL + endpoint + '?token=' + token
    return requests.get(url)


def api_post(endpoint, token, payload):
    BASE_URL = 'https://gdex.ucar.edu/api/'
    url = BASE_URL + endpoint + '?token=' + token
    return requests.post(url, json=payload)


if __name__ == "__main__":
    raise SystemExit(main())
