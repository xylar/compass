import numpy as np
import xarray as xr


def fill_zlevel_bathymetry_holes(ds):
    """
    Modifies ``maxLevelCell`` and ``bottomDepth`` in the given dataset to
    fill holes in the bathymetry where a given cell has layers that are deeper
    than any of its neighbors.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset containing ``maxLevelCell``, ``bottomDepth``, ``cellsOnCell``
        and ``refBottomDepth``
    """
    max_level_cell = ds.maxLevelCell - 1
    bottom_depth = ds.bottomDepth
    ref_bottom_depth = ds.refBottomDepth
    coc = ds.cellsOnCell - 1
    ncells = ds.sizes['nCells']
    max_edges = ds.sizes['maxEdges']

    deepest_neighbor = np.zeros(ncells, dtype=int)
    for i in range(max_edges):
        mask = coc[:, i] >= 0
        neighbor_max_level = max_level_cell[coc[:, i]]
        deepest_neighbor[mask] = np.maximum(deepest_neighbor[mask],
                                            neighbor_max_level[mask])

    deepest_neighbor = xr.DataArray(dims=('nCells',), data=deepest_neighbor)
    bottom_depth_neighbor = ref_bottom_depth[deepest_neighbor]
    holes = max_level_cell <= deepest_neighbor
    max_level_cell = xr.where(holes, deepest_neighbor, max_level_cell)
    bottom_depth = xr.where(holes, bottom_depth_neighbor, bottom_depth)

    ds['maxLevelCell'] = max_level_cell + 1
    ds['bottomDepth'] = bottom_depth
