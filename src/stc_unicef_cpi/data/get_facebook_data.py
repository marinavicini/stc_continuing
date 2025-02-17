"""GET DELIVERY ESTIMATES FROM FACEBOOK MARKETING API"""

import time
import warnings

import pandas as pd

from stc_unicef_cpi.utils.constants import opt
from stc_unicef_cpi.utils.general import get_facebook_credentials
from stc_unicef_cpi.utils.geospatial import get_hex_radius

try:
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.api import FacebookAdsApi
except ImportError:
    warnings.warn(
        "Necessary modules for facebook data not installed - assumed not desired"
    )


def fb_api_init(token, id):
    """Init Facebook API

    :param token: Access token
    :type access_token: str
    :param id: Account id
    :type ad_account_id: str
    :return: api and account connection
    :rtype: conn
    """
    api = FacebookAdsApi.init(access_token=token)
    account = AdAccount(id)
    try:
        account.get_ads()
        print("Initialized successfully!")
    except Exception as e:
        if e._api_error_code == 190:
            raise ValueError("Invalid or expired access token!")
        elif e._api_error_code == 100:
            raise ValueError("Invalid ad account id!")
        else:
            raise RuntimeError("Please check you credentials!")

    return api, account


def define_params(lat, lon, radius, opt):
    """Define search parameters

    :param lat: latitude
    :type lat: str
    :param long: longitude
    :type long: str
    :param radius: radius
    :type radius: float
    :param opt: optimization criteria
    :type opt: string
    """
    geo = {
        "latitude": lat,
        "longitude": lon,
        "radius": radius,
        "distance_unit": "kilometer",
    }
    targeting = {
        "geo_locations": {
            "custom_locations": [
                geo,
            ],
        },
    }
    params = {
        "optimization_goal": opt,
        "targeting_spec": targeting,
    }

    return params


def point_delivery_estimate(account, lat, lon, radius, opt):
    """Point delivery estimate
    :return: _description_
    :rtype: _type_
    """
    params = define_params(lat, lon, radius, opt)
    d_e = account.get_delivery_estimate(params=params)
    delivery_estimate = pd.DataFrame(d_e)
    return delivery_estimate


def delivery_estimate(account, lat, long, radius, opt):
    row = point_delivery_estimate(account, lat, long, radius, opt)
    row["lat"], row["long"] = lat, long
    return row


def get_facebook_estimates(coords, out_dir, name_out, res):
    """Get delivery estimates from a lists of coordinates

    :return:
    :rtype:
    """
    token, account_id = get_facebook_credentials("../../../conf/credentials.yaml")
    data = pd.DataFrame()
    _, account = fb_api_init(token, account_id)
    radius = get_hex_radius(res)
    for i, (lat, long) in enumerate(coords):
        try:
            row = delivery_estimate(account, lat, long, radius, opt)
            data = data.append(row, ignore_index=True)
        except Exception as e:
            if e._api_error_code == 80004:
                print(f"Too many calls!\nStopped at {i}, ({lat, long}).")
                data.to_parquet(f"{out_dir}/{name_out}")
                time.sleep(3800)
                row = delivery_estimate(account, lat, long, radius, opt)
            else:
                print(f"Point {i}, ({lat, long}) not found.")
                row = pd.DataFrame()
                row["lat"], row["long"] = lat, long
            data = data.append(row, ignore_index=True)
    data["hex_centroid"] = coords
    data.to_parquet(f"{out_dir}/{name_out}")
    return data
