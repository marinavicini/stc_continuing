import math
from itertools import chain

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
import shapely.wkt
from pyproj import Geod
from shapely import geometry, wkt
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import pycountry
from pyquadkey2 import quadkey as qk
import dask.dataframe as dd

import stc_unicef_cpi.utils.clean_text as ct
import stc_unicef_cpi.utils.general as g

# resolution and area of hexagon in km2
res_area = {
    0: 4250546.8477000,
    1: 607220.9782429,
    2: 86745.8540347,
    3: 12392.2648621,
    4: 1770.3235517,
    5: 252.9033645,
    6: 36.1290521,
    7: 5.1612932,
    8: 0.7373276,
    9: 0.1053325,
    10: 0.0150475,
    11: 0.0021496,
    12: 0.0003071,
    13: 0.0000439,
    14: 0.0000063,
    15: 0.0000009,
}


def get_hex_radius(res):
    """Get radius according to h3 resolution
    :param res: resolution
    :type res: int
    :return: radius corresponding to the resolution
    :rtype: float
    """
    for key, value in res_area.items():
        if key == res:
            radius = math.sqrt(value * 2 / (3 * math.sqrt(3)))
    return radius


def get_lat_long(data, geo_col):
    """Get latitude and longitude points
    from a given geometry column
    :param data: dataset
    :type data: dataframe
    :param geo_col: name of column containing the geometry
    :type geo_col: string
    :return: dataset
    :rtype: dataframe with latitude and longitude columns
    """
    data["lat"] = data[geo_col].map(lambda p: p.x)
    data["long"] = data[geo_col].map(lambda p: p.y)
    return data


def get_hex_centroid(data, hex_code="hex_code"):
    """Get centroid of hexagon
    :param data: dataset
    :type data: dataframe
    :param hex_code: name of column containing the hexagon code
    :type hex_code: string
    :return: coords
    :rtype: list of tuples
    """
    data["hex_centroid"] = data[[hex_code]].apply(
        lambda row: h3.h3_to_geo(row[hex_code]), axis=1
    )
    data["lat"], data["long"] = data["hex_centroid"].str
    return data


def create_geometry(data, lat, long):
    """Create geometry column from longitude (x) and latitude (y) columns
    :param data: dataset
    :type data: dataframe
    :param lat: name of column containing the longitude of a point
    :type lat: string
    :param long: name of column containing the longitude of a point
    :type long: string
    :return: data
    :rtype: datafrane with geometry column
    """
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[long], data[lat]))
    return data


def get_hex_code(df, lat, long, res):
    df["hex_code"] = df[[lat, long]].swifter.apply(
        lambda row: h3.geo_to_h3(row[lat], row[long], res), axis=1
    )
    return df

def get_hex_code_w_dask(data, lat, lon, resolution):
    # NB ideal to have partitions around 100MB in size
    # client.restart()
    ddf = dd.from_pandas(
        data,
        npartitions=max(
        [4, int(data.memory_usage(deep=True).sum() // int(1e8))]
        ),
    )  # chunksize = max_records(?)
    print(f"Using {ddf.npartitions} partitions")
    ddf["hex_code"] = ddf[[lat, lon]].apply(
        lambda row: h3.geo_to_h3(
            row[lat], row[lon], resolution
        ),
        axis=1,
        meta=(None, int),
    )
    # ddf = ddf.drop(columns=[lat, lon])
    # print("Done!")
    # print("Aggregating within cells...")
    # ddf = ddf.groupby("hex_code").agg(
    #     {col: agg_fn for col in ddf.columns if col != "hex_code"}
    #     )
    # data = ddf.compute() 
    return ddf

# def aggregate_hexagon_fn(df, agg_fn):
#     df = df.groupby(by=["hex_code"]).agg(
#         {col: agg_fn for col in df.columns if col != "hex_code"}
#     )
#     return df

def aggregate_hexagon_fn(df, agg_dic):
    cols = list(df.columns)
    agg_dic = g.subset_dic(cols, agg_dic)
    df = df.groupby(by=["hex_code"]).agg(agg_dic).reset_index()
    return df


def aggregate_hexagon(df, col_to_agg, name_agg, type):
    if type == "count":
        df = df.groupby("hex_code", as_index=False).count()
    else:
        df = df.groupby("hex_code", as_index=False).mean()
    df = df[["hex_code", col_to_agg]]
    df = df.rename({col_to_agg: name_agg}, axis=1)
    return df


def get_shape_for_ctry(ctry_name):
    # world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # ctry_shp = world[world.name == ctry_name]
    ctry_code = ct.get_alpha3_code(ctry_name)
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    world = reader.records()
    # print(f'country name: {ctry_name}, country code: {ctry_code}')
    try:
        # ctry_shp = next(filter(lambda x: x.attributes["NAME"] == ctry_name, world)).geometry
        ctry_shp = next(filter(lambda x: x.attributes["ADM0_ISO"] == ctry_code, world)).geometry
    except StopIteration:    
        world = reader.records()
        ctry_shp = next(filter(lambda x: x.attributes["ADM0_A3"] == ctry_code, world)).geometry

    return ctry_shp


def get_hexes_for_ctry(ctry_name="Nigeria", res=7):
    """Get array of all hex codes for specified country
    :param ctry_name: _description_, defaults to 'Nigeria'
    :type ctry_name: str, optional
    :param level: _description_, defaults to 7
    :type level: int, optional
    """
    ctry_shp = get_shape_for_ctry(ctry_name)
    
    try:
        # handle MultiPolygon
        ctry_polys = list(ctry_shp)
        hexes = [
            h3.polyfill(poly.__geo_interface__, res, geo_json_conformant=True)
            # h3.polyfill(poly.__geo_interface__, res)
            for poly in ctry_polys
        ]
        return np.array(list(chain.from_iterable(hexes)), dtype=int)

    except TypeError:
        # only normal Polygon
        ctry_shp = ctry_shp.__geo_interface__
        return h3.polyfill(ctry_shp, res, geo_json_conformant=True)
        # return h3.polyfill(ctry_shp, res)


def get_new_nbrs_at_k(hexes, k):
    """Given set of hexes, return set of new neighbors at k distance
    Useful for expanding country hex sets

    :param hexes: _description_
    :type hexes: _type_
    :param k: _description_
    :type k: _type_
    """

    def hex_nbrs_at_k(hex, k):
        return np.array(
            list(
                chain.from_iterable(
                    [h3.hex_ring(hex, dist) for dist in range(1, k + 1)]
                )
            ),
            dtype=int,
        )

    nbrs_at_k = pd.DataFrame(hexes, columns=["hex_code"]).hex_code.swifter.apply(
        lambda hex: hex_nbrs_at_k(hex, k)
    )
    return np.setdiff1d(np.array(list(chain.from_iterable(nbrs_at_k))), hexes)


def get_area_polygon(polygon, crs="WGS84"):
    """Get area of a polygon on earth in km squared
    :param polygon: Polygon
    :type polygon: Polygon
    """
    geometry = wkt.loads(str(polygon))
    geod = Geod(ellps=crs)
    area = geod.geometry_area_perimeter(geometry)[0]
    area = area / 10**6
    return area


def get_poly_boundary(df, hex_code):
    df["geometry"] = [
        Polygon(h3.h3_to_geo_boundary(x, geo_json=True)) for x in df[hex_code]
    ]
    return df


def format_polygons(poly):
    """Retrive type of polygon and convert to list"""
    geom = shapely.wkt.loads(poly)
    if geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    elif geom.geom_type == "Polygon":
        polygons = [geom]
    else:
        raise OSError("Shape is not a polygon.")
    return polygons


def hexes_poly(poly, res):
    """Get dataframe with hexagons belonging to a polygon
    :param poly: polygon
    :type poly: polygon
    :param res: resolution
    :type res: int
    :raises IOError: _description_
    :return: _description_
    :rtype: _type_
    """
    polygons = format_polygons(poly)
    hexs = [
        h3.polyfill(geometry.mapping(polygon), res, geo_json_conformant=True)
        for polygon in polygons
    ]
    hexs = list({item for sublist in hexs for item in sublist})
    df = pd.DataFrame(hexs)
    df.rename(columns={0: "hex_code"}, inplace=True)
    df["geometry"] = poly
    return df

def get_quadkey_polygon(qkey):
    '''Get polygon associated with quadkey'''
    # save string as quadkey
    qkey = qk.QuadKey(str(qkey))
    # get coordinates
    n, w = qkey.to_geo(0)
    s, e = qkey.to_geo(3)

    poly_quadkey = Polygon(
        [Point([w, n]), Point([e, n]), Point([e, s]), Point([w, s])]
    )
    return poly_quadkey