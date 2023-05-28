import mpas_tools.mesh.creation.mesh_definition_tools as mdt
import numpy as np
from geometric_features import read_feature_collection
from mpas_tools.cime.constants import constants
from mpas_tools.mesh.creation.signed_distance import (
    signed_distance_from_geojson,
)

from compass.mesh import QuasiUniformSphericalMeshStep


class YAM10to60BaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating YAM10to60 meshes
    """
    def setup(self):
        """
        Add some input files
        """
        package = 'compass.ocean.tests.global_ocean.mesh.yam10to60'
        self.add_input_file(filename='northern_south_atlantic.geojson',
                            package=package)
        self.add_input_file(filename='amazon_delta.geojson',
                            package=package)
        super().setup()

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """

        dlon = 0.1
        dlat = dlon
        nlon = int(360. / dlon) + 1
        nlat = int(180. / dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        cell_width_vs_lat = mdt.EC_CellWidthVsLat(lat)
        cell_width = np.outer(cell_width_vs_lat, np.ones([1, lon.size]))

        # read the shape
        fc = read_feature_collection('northern_south_atlantic.geojson')

        # How wide in meters the smooth transition between the background
        #   resolution and the finer resolution regions should be.
        # 1200 km is equivalent to about 10 degrees latitude
        trans_width = 1200e3

        # The resolution in km of the finer resolution region
        fine_cell_width = 20.

        # the radius of the earth defined in E3SM's shared constants
        earth_radius = constants['SHR_CONST_REARTH']

        # A field defined on the lat-long grid with the signed distance away
        # from the boundary of the shape (positive outside and negative inside)
        atlantic_signed_distance = signed_distance_from_geojson(
            fc, lon, lat, earth_radius, max_length=0.25)

        # A field that goes smoothly from zero inside the shape to one outside
        # the shape over the given transition width.
        weights = 0.5 * (1 + np.tanh(atlantic_signed_distance / trans_width))

        # The cell width in km becomes a blend of the background cell width
        # and the finer cell width using the weights
        cell_width = fine_cell_width * (1 - weights) + cell_width * weights

        # read the shape
        fc = read_feature_collection('amazon_delta.geojson')

        # 400 km is equivalent to about 3 degrees latitude
        trans_width = 400e3

        # The resolution in km of the finer resolution region
        fine_cell_width = 10.

        # A field defined on the lat-long grid with the signed distance away
        # from the boundary of the shape (positive outside and negative inside)
        amazon_delta_signed_distance = signed_distance_from_geojson(
            fc, lon, lat, earth_radius, max_length=0.25)

        # A field that goes smoothly from zero inside the shape to one outside
        # the shape over the given transition width.
        weights = 0.5 * (1 + np.tanh(
            amazon_delta_signed_distance / trans_width))

        # The cell width in km becomes a blend of the background cell width
        # and the finer cell width using the weights
        cell_width = fine_cell_width * (1 - weights) + cell_width * weights

        return cell_width, lon, lat
