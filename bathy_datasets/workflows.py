"""
A placeholeder for the prototype workflows used for generating the data
for the ARDC-GMRT project.
"""

from typing import Any, Dict, List, Tuple
import geopandas
import tiledb.cloud
from tiledb.cloud.compute import Delayed

from bathy_datasets import rhealpix, utils


def geometry_slices(gdf: geopandas.GeoDataFrame) -> List[Tuple[slice, slice]]:
    """
    Given a dataframe of rhealpix codes, retrieve the bounding box for each
    code and return a list of tuples containing slice objects to access
    portions of a TileDB array.
    """
    slices = []

    for geom in rhealpix.rhealpix_geo_boundary(
        gdf.region_code.values, round_coords=False
    ):
        bounds = geom.bounds
        slices.append((slice(bounds[0], bounds[-2]), slice(bounds[1], bounds[-1])))

    return slices


def gather_stats(results):
    """
    Gather the results from all the stats tasks and
    combine into a single dict.
    Des
    """
    data = {}

    for item in results:
        for key in item:
            data[key] = item[key]

    return data


def dataset_statistics(
    gdf: geopandas.GeoDataFrame,
    array_uri: str,
    tiledb_config: Dict[str, Any],
    n_partitions: int = 12,
    n_sub_partitions: int = 3,
) -> Delayed:
    """
    Produce the task tree for calculating the dataset level statistics.
    It is exectued in a scatter -> reduce fashion as the stats are
    evaluated recursively.
    Assumes the user has already logged in to their tiledb.cloud account.
    """
    reduce_tasks = []

    ctx = tiledb.Ctx(config=tiledb_config)

    with tiledb.open(array_uri, ctx=ctx) as ds:
        attributes = [ds.attr(i).name for i in range(ds.nattr)]
        dims = [ds.dim(i).name for i in range(ds.ndim)]

    for dim in dims[::-1]:  # insert last to first
        attributes.insert(0, dim)

    tasks_dict = {stat: [] for stat in attributes}

    slices = geometry_slices(gdf)
    blocks = utils.scatter(slices, n_partitions)

    # build the scatter part DAG
    for i, block in enumerate(blocks):
        sub_blocks = utils.scatter(block, n_sub_partitions)

        for si, sub_block in enumerate(sub_blocks):
            for attribute in attributes:
                if attribute in ["X", "Y"]:
                    schema = attribute
                else:
                    schema = None

                task_name = f"block-{i}-sub_block-{si}-{attribute}"
                task = Delayed("sixy6e/basic_statistics_incremental", name=task_name)(
                    array_uri,
                    tiledb_config,
                    attribute,
                    schema=schema,
                    idxs=sub_block,
                    summarise=False,
                )

                # establishing a dependency tree
                # (no real reason to do other than not to spin up 50000 tasks)
                if len(tasks_dict[attribute]) > 1:
                    task.depends_on(tasks_dict[attribute][-1])

                tasks_dict[attribute].append(task)

    # build the reduce part of the DAG
    for attribute in attributes:
        task_name = f"reduce-attibute-{attribute}"
        reducer_task = Delayed("sixy6e/basic_statistics_reduce", name=task_name)(
            tasks_dict[attribute], attribute
        )
        reduce_tasks.append(reducer_task)

    collect_stats_task = Delayed(gather_stats, local=True, name="gather-stats")(
        reduce_tasks
    )

    return collect_stats_task
