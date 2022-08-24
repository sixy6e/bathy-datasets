import numpy
from numba import jit
from scipy import stats
import tiledb
import tiledb.cloud


def basic_statistics(data, attribute):
    """
    Calculate basic statistics.
    Stats calculated:
        * minimum
        * maximum
        * count
        * total/sum
        * mean
        * variance
        * standard deviation
        * skewness
        * kurtosis
    """
    data_attr = data[attribute]

    result = {
        attribute: {
            "minimum": numpy.nanmin(data_attr),
            "maximum": numpy.nanmax(data_attr),
            "count": data_attr.shape[0],
            "total": numpy.nansum(data_attr),
            "mean": numpy.nanmean(data_attr),
            "variance": numpy.nanvar(data_attr, ddof=1),  # unbiased sample
            "stddev": numpy.nanstd(data_attr, ddof=1),  # unbiased sample
            "skewness": stats.skew(data_attr, nan_policy="omit"),
            "kurtosis": stats.kurtosis(data_attr, nan_policy="omit"),
        }
    }

    return result


def reduce(results_list, attribute):
    """
    The reducer for the incremental basic statistics routine.
    Tiles/blocks/chunks of data can be distributed across a bunch of
    workers. This routine takes all of those individual results, and
    reduces them to generate the basic statistiscs.
    See for method:
    https://math.stackexchange.com/questions/1765042/moving-window-computation-of-skewness-and-kurtosis
    """
    gather = {
        "minimum": [],
        "maximum": [],
        "count": [],
        "total": [],
        "mu2_prime": [],
        "mu3_prime": [],
        "mu4_prime": [],
    }

    for res in results_list:
        for stat in res[attribute]:
            gather[stat].append(res[attribute][stat])

    minv = numpy.min(gather["minimum"])
    maxv = numpy.max(gather["maximum"])
    total = numpy.sum(gather["total"])
    count = numpy.sum(gather["count"])
    mu2_prime = numpy.sum(gather["mu2_prime"])
    mu3_prime = numpy.sum(gather["mu3_prime"])
    mu4_prime = numpy.sum(gather["mu4_prime"])

    xbar = total / count
    mu2_prime = mu2_prime / count
    mu3_prime = mu3_prime / count
    mu4_prime = mu4_prime / count

    mu2 = mu2_prime - xbar**2
    mu3 = mu3_prime - 3 * xbar * mu2_prime + 2 * xbar**3
    mu4 = mu4_prime - 4 * xbar * mu3_prime + 6 * xbar**2 * mu2_prime - 3 * xbar**4

    sigma2 = mu2 * count / (count - 1)
    sigma = numpy.sqrt(sigma2)
    skew = mu3 / numpy.sqrt(mu2**3)
    kurt = (mu4 / mu2**2) - 3
        
    result = {
        attribute: {
            "minimum": minv,
            "maximum": maxv,
            "count": count,
            "total": total,
            "mean": xbar,
            "variance": sigma2,
            "stddev": sigma,
            "skewness": skew,
            "kurtosis": kurt,
        }
    }

    return result


@jit(nopython=True)
def _increment_stats(data: numpy.ndarray, double: bool):
    """
    Incremental gathering of required stats to calculate the 2nd, 3rd and 4th
    moments.
    NaN's are ignored.
    """
    n = 0
    if double:
        total = 0.0
        mu2_prime = 0.0
        mu3_prime = 0.0
        mu4_prime = 0.0
    else:
        total = 0
        mu2_prime = 0
        mu3_prime = 0
        mu4_prime = 0

    minv = data[0]
    maxv = data[0]

    for i in data:

        if numpy.isnan(i):
            continue

        n += 1
        total += i
        mu2_prime += i**2
        mu3_prime += i**3
        mu4_prime += i**4

        minv = min(minv, i)
        maxv = max(maxv, i)

    return n, total, mu2_prime, mu3_prime, mu4_prime, minv, maxv


def basic_statistics_incremental(
    array_uri, config, attribute, schema=None, idxs=slice(None), summarise=True
):
    """
    An incremental version of basic_statistics so that large arrays
    can be processed.
    Using a single pass algorithm, see:
    https://math.stackexchange.com/questions/1765042/moving-window-computation-of-skewness-and-kurtosis

    Stats calculated:
        * minimum
        * maximum
        * count
        * total
        * mean
        * variance
        * standard deviation
        * skewness
        * kurtosis
    """
    attrs = [attribute]
    coords = False
    if schema:
        # calculate stats on the given schema dimension
        attribute = schema
        attrs = ["Z"]
        coords = True

    mapper = {
        "f": True,
        "u": False,
        "i": False,
    }

    with tiledb.open(array_uri, config=config) as ds:
        if schema:
            kind = ds.dim(schema).dtype.kind
        else:
            kind = ds.schema.attr(attribute).dtype.kind

        double = mapper.get(kind, None)
        if double is None:
            raise TypeError("Attibute must be of type float or integer")

        n = 0

        if double:
            total = 0.0
            mu2_prime = 0.0
            mu3_prime = 0.0
            mu4_prime = 0.0
        else:
            total = 0
            mu2_prime = 0
            mu3_prime = 0
            mu4_prime = 0

        minv = []
        maxv = []

        for idx in idxs:
            query = ds.query(attrs=attrs, coords=coords, return_incomplete=True).multi_index[idx]

            for tile in query:
                interim_result = _increment_stats(tile[attribute], double)
                n += interim_result[0]
                total += interim_result[1]
                mu2_prime += interim_result[2]
                mu3_prime += interim_result[3]
                mu4_prime += interim_result[4]
                minv.append(interim_result[5])
                maxv.append(interim_result[6])

                tile[attribute] = None

    minv = numpy.min(minv)
    maxv = numpy.max(maxv)

    result = {
        attribute: {
           "minimum": minv,
           "maximum": maxv,
           "count": n,
           "total": total,
           "mu2_prime": mu2_prime,
           "mu3_prime": mu3_prime,
           "mu4_prime": mu4_prime,
        }
    }

    if summarise:
        result = reduce([result], attribute)

    return result


token = ""
tiledb.cloud.login(token=token)
tiledb.cloud.udf.register_generic_udf(
    basic_statistics, "basic_statistics", namespace="AusSeabed"
)
tiledb.cloud.udf.register_generic_udf(
    basic_statistics_incremental, "basic_statistics_incremental", namespace="AusSeabed"
)
tiledb.cloud.udf.register_generic_udf(
    reduce, "basic_statistics_reduce", namespace="AusSeabed"
)
