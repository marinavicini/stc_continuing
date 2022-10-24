import pandas as pd
import unidecode
from pathlib import Path
import pycountry
import pycountry_convert as pc
import re

import stc_unicef_cpi.utils.constants as c
import stc_unicef_cpi.utils.general as g

def remove_accent(acc_string: str) -> str:
    '''Return string without accents'''
    unaccented_string = unidecode.unidecode(acc_string)
    return unaccented_string

def remove_apostrophe(string: str) -> str:
    return re.sub("'", "", string)

def format_country_name(to_clean: str) -> str:
    to_clean = remove_accent(to_clean)
    to_clean = remove_apostrophe(to_clean)
    return to_clean


def get_country_name_gaul(name):
    '''Get GAUL name of country based on a dictionary I created in constants
    To be removed'''
    if name in c.dic_pycountry_to_gaul.keys():
        return c.dic_pycountry_to_gaul[name]
    else:
        return name


def download_country_codes(out_dir):
    _out_dir = Path(out_dir)

    if not (_out_dir / "country_codes.csv").exists():
        print(" -- Retrieving country codes")
        codes_url = c.url_country_codes
        out_cc = _out_dir / "country_codes.csv"
        g.download_file(codes_url, out_cc)


def iso_to_gaul_code(code_iso, out_dir):
    # https://datahub.io/core/country-codes#python
    download_country_codes(out_dir)
    country_codes = pd.read_csv(Path(out_dir) / "country_codes.csv")

    d = dict(zip(country_codes['ISO3166-1-Alpha-3'], country_codes['GAUL']))
    try:
        return int(d[code_iso])
    except:
        raise ValueError('Wrong Alpha 3 ISO code')
    
def get_alpha3_code(country):
    '''Alpha 3 code (three letters iso code'''
    country_code = pycountry.countries.get(name = country).alpha_3
    return country_code


def iso3_to_continent_name(iso):
    '''From a country iso code return continent name'''
    iso_2 = pc.country_alpha3_to_country_alpha2(iso)
    continent_code = pc.country_alpha2_to_continent_code(iso_2)
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    return continent_name


def get_commuting_continent(iso):
    '''Get Continent name as written in the commuting '''
    continent_name = iso3_to_continent_name(iso)
    commuting_continent_names = {'North America':'North', 'South America':'South'}
    
    if continent_name in commuting_continent_names.keys():
        return commuting_continent_names[continent_name]
    else:
        return continent_name