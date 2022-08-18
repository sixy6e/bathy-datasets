"""
A placeholeder for the prototype workflows used for generating the data
for the ARDC-GMRT project.
"""

from typing import Any, Dict, List, Tuple
import json
from pathlib import Path
import numpy
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


def dummy_reducer(results):
    """
    A dummy reducer for a task DAG. Essentially define an entrypoint
    to the whole DAG.
    """

    return len(results)


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

    tasks_dict: Dict[str, List[Any]] = {stat: [] for stat in attributes}

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


def convert_gsf_ping_beam(
    files: List[str],
    array_uris: List[str],
    cell_freq_uris: List[str],
    ctx: tiledb.Ctx,
    processing_node_limit: int = 40,
    ping_slice_step: int = 2000,
    slices_per_node: int = 3,
    ping_beam_dims: bool = False,
):
    """
    Prototype GSF -> TileDB converter using Ping and Beam numbers as the dimensional axes.
    This workflow produces a 1-1 relationship between GSF and TileDB arrays; unless
    the GSF contains no pings.
    The TileDB arrays are 2-Dimensional of the order array[ping, beam] axes.
    The resource class is hardcoded to be large, which is 8GB and 8CPUs.
    The task tree also sets up to calculate the cell frequency once all pings
    have been ingested.
    The processing node count was a kind of limiter on how many tasks to run in
    parallel.
    """
    node_counter = 0
    tasks = []
    tasks_dict: Dict[int, List[Any]] = {n: [] for n in range(processing_node_limit)}
    files_dict: Dict[str, List[Any]] = {fname: [] for fname in files}
    cell_frequency_tasks = []

    vfs = tiledb.VFS(ctx=ctx)
    config = ctx.config()

    for enumi, pathname in enumerate(files):
        metadata_pathname = pathname.replace(".gsf", ".json")
        base_name = Path(pathname).stem

        with vfs.open(metadata_pathname) as src:
            gsf_metadata = json.loads(src.read())

        ping_count = gsf_metadata["file_record_types"]["GSF_SWATH_BATHYMETRY_PING"][
            "record_count"
        ]

        # ideally filter the empty GSFs prior, but still good to check
        if not ping_count:
            # no pings, skipping
            continue

        # split tasks into groups of pings
        slices = [
            slice(start, start + ping_slice_step)
            for start in numpy.arange(0, ping_count, ping_slice_step)
        ]
        slice_chunks = [
            slices[i : i + slices_per_node]
            for i in range(0, len(slices), slices_per_node)
        ]

        array_uri = array_uris[enumi]
        cell_freq_uri = cell_freq_uris[enumi]

        for slice_chunk in slice_chunks:
            start_idx = slice_chunk[0].start
            end_idx = slice_chunk[-1].stop
            task_name = f"{base_name}-{start_idx}-{end_idx}-{node_counter}"

            task = Delayed(
                "sixy6e/ingest_gsf_slices",
                name=task_name,
                image_name="3.7-geo",  # might be no longer required
                timeout=1800,
                resource_class="large",  # not tested; wasn't avail during the project
            )(
                gsf_metadata["gsf_uri"],
                config["vfs.s3.aws_access_key_id"],
                config["vfs.s3.aws_secret_access_key"],
                array_uri,
                slice_chunk,
                cell_frequency=False,
                ping_beam_dims=ping_beam_dims,
            )

            # setting up a false dependency (reduce the number of tasks running at once)
            if len(tasks_dict[node_counter]):
                task.depends_on(tasks_dict[node_counter][-1])

            tasks.append(task)
            tasks_dict[node_counter].append(task)
            node_counter += 1

            files_dict[pathname].append(task)

            if node_counter == processing_node_limit:
                node_counter = 0

        # define the cell frequency part of the DAG for the current GSF DAG
        cell_freq_task = Delayed(
            "sixy6e/cell_frequency", name=f"{base_name}-cell-frequency"
        )(
            array_uri,
            cell_freq_uri,
            config["vfs.s3.aws_access_key_id"],
            config["vfs.s3.aws_secret_access_key"],
        )

        # all pings for this GSF must be done before doing the cell freq
        for dep in files_dict[pathname]:
            cell_freq_task.depends_on(dep)

        cell_frequency_tasks.append(cell_freq_task)

    reduce_task = Delayed(dummy_reducer, name="dummy-reducer", local=True)(
        cell_frequency_tasks
    )

    return reduce_task
