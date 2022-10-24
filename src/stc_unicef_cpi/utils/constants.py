import inspect
from pathlib import Path
import pandas as pd

# optimization objective for facebook audience estimates
opt = "REACH"

# names for data retrieved through open cell id
open_cell_colnames = [
    "radio",
    "mcc",
    "mnc",
    "lac",
    "cid",
    "range",
    "long",
    "lat",
    "sample",
    "changeable_1",
    "changeable_0",
    "created",
    "updated",
    "avg_signal",
]

# google earth engine parameters
start_ee = "2010-01-01"
end_ee = "2020-01-01"
res_ee = 500
folder_ee = "gee"

current_dir = Path.cwd()
if current_dir.name == "data" and current_dir.parent.name == "stc_unicef_cpi":
    # restrict to when calling from make_dataset.py
    # base directory for data
    base_dir_data = Path.cwd().parent.parent.parent / "data"
    # base directory for autoencoder models
    base_dir_model = Path.cwd().parent.parent.parent / "models"
    # external data
    ext_data = base_dir_data / "external"
    ext_data.mkdir(exist_ok=True)

    # interim data
    int_data = base_dir_data / "interim"
    int_data.mkdir(exist_ok=True)

    # processed data
    proc_data = base_dir_data / "processed"
    proc_data.mkdir(exist_ok=True)

    # raw data
    raw_data = base_dir_data / "raw"
    raw_data.mkdir(exist_ok=True)

    # tiff files
    tiff_data = ext_data / "tiff"
    tiff_data.mkdir(exist_ok=True)

else:
    # just make objects none so no import errors
    # true if not importing from notebook
    base_dir_data = None  # type: ignore
    # base directory for autoencoder models
    base_dir_model = None  # type: ignore
    # external data
    ext_data = None  # type: ignore
    # interim data
    int_data = None  # type: ignore
    # processed data
    proc_data = None  # type: ignore
    # raw data
    raw_data = None  # type: ignore
    # tiff files
    tiff_data = None  # type: ignore
    # importing_file = Path(__file__).name
    # if importing_file == "make_dataset.py":
    # base_dir_data = current_dir / "data"
    # base_dir_model = current_dir / "models"
    # base_dir_data.mkdir(exist_ok=True)
    # base_dir_model.mkdir(exist_ok=True)
    # raise ValueError(
    #     "Must run make_dataset.py from stc_unicef_cpi/data directly for default paths to work as intended: constants.py should only be relevant for this script."
    # )


# loggers
str_log = "data_streamer"
dataset_log = "make_dataset.log"

# variables

cols_commuting = [
    "hex_code",
    "name_commuting",
    "win_population_commuting",
    "win_roads_km_commuting",
    "area_commuting",
]

# threshold
cutoff = 30

# speedtest params
serv_type = "mobile"
serv_year = 2021
serv_quart = 4

# countries in Sub-saharan africa
countries_ssf = ['Angola', 
            'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
            'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo, The Democratic Republic of the', 'Congo', "Côte d'Ivoire",
            'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 
            'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau',
            'Kenya', 
            'Lesotho', 'Liberia', 
            'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique',
            'Namibia', 'Niger', 'Nigeria',
            'Rwanda', 
            'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'Sudan', 'South Sudan', 
            'Tanzania, United Republic of', 'Togo', 
            'Uganda',
            'Zambia', 'Zimbabwe']


countries_code_ssf = ['AGO',
                    'BEN', 'BWA', 'BFA', 'BDI',
                    'CAF', 'CIV', 'CMR', 'COD', 'COG', 'COM', 'CPV', 
                    'ERI', 'ETH', 
                    'GAB', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ',
                    'KEN',
                    'LBR', 'LSO', 
                    'MDG', 'MLI', 'MOZ', 'MRT', 'MUS', 'MWI',
                    'NAM', 'NER', 'NGA', 
                    'RWA',
                    'SDN', 'SEN', 'SLE', 'SOM', 'SSD', 'STP', 'SWZ', 'SYC', 
                    'TCD', 'TGO', 'TZA', 
                    'UGA', 
                    'ZAF', 'ZMB', 'ZWE']


# countries name that not match between pycountry library and gaul name
# key -> pycountry version
# value -> gaul version
dic_pycountry_to_gaul = {'Cabo Verde':'Cape Verde',
    'Tanzania, United Republic of': 'United Republic of Tanzania',
    'Congo, The Democratic Republic of the':'Democratic Republic of the Congo',
    'Eswatini':'Swaziland'}


agg_fn = {
    # copland
    'elevation':'mean',
    'evapotrans':'mean',
    'built':pd.Series.mode,
    'cnfd':'mean',
    'accessibility':'median',
    'accessibility_walking_only':'median',
    'ndvi':'mean',
    'ndwi':'mean',
    'avg_rad':'mean',
    'cf_cvg':'mean',
    'pdsi':'mean',
    'Optical_Depth_047':'mean',
    'Optical_Depth_055':'mean',
    'precipiacc':'sum',
    'precimean':'mean',
    'precistd':'mean',
    'slope':'mean',
    'population':'sum',
    'M_0':'sum',
    'M_1':'sum',
    'M_5':'sum',
    'M_10':'sum',
    'M_15':'sum',
    'M_30':'sum',
    'M_20':'sum',
    'M_25':'sum',
    'M_35':'sum',
    'M_40':'sum',
    'M_45':'sum',
    'M_50':'sum',
    'M_55':'sum',
    'M_60':'sum',
    'M_70':'sum',
    'M_75':'sum',
    'M_65':'sum',
    'M_80':'sum',
    'F_0':'sum',
    'F_1':'sum',
    'F_5':'sum',
    'F_10':'sum',
    'F_15':'sum',
    'F_20':'sum',
    'F_25':'sum',
    'F_30':'sum',
    'F_35':'sum',
    'F_40':'sum',
    'F_45':'sum',
    'F_60':'sum',
    'F_50':'sum',
    'F_55':'sum',
    'F_65':'sum',
    'F_70':'sum',
    'F_75':'sum',
    'F_80':'sum',
    '2019gdp':'mean',
    'africa':'mean',
    'ec2019':'mean',
    'GDP_PPP_1990':'mean',
    'GDP_PPP_2000':'mean',
    'GDP_PPP_2015':'mean'
}



# URL
# conflicts, version 2022
url_conflict = "https://ucdp.uu.se/downloads/ged/ged221-csv.zip"
# infrastructure, jun 2021
url_infrastructure = "https://zenodo.org/record/4957647/files/CISI.zip?download=1"
# Real GDP
url_real_gdp = "https://figshare.com/ndownloader/files/31456837"
# GDP PPP
url_gdp_ppp = "https://datadryad.org/stash/downloads/file_stream/241958"
# Electric consumption
url_elec_cons = "https://figshare.com/ndownloader/files/31456843"
# commuting, 
# url_commuting_zones = "https://data.humdata.org/dataset/b7aaa3d7-cca2-4364-b7ce-afe3134194a2/resource/37c2353d-08a6-4cc2-8364-5fd9f42d6b64/download/data-for-good-at-meta-commuting-zones-july-2021.csv"
url_commuting_zones = "https://data.humdata.org/dataset/b7aaa3d7-cca2-4364-b7ce-afe3134194a2/resource/3c068b51-5f0d-4ead-80ba-97312ec034e4/download/data-for-good-at-meta-commuting-zones-august-2022.csv.csv"


# Url country codes
url_country_codes = 'https://pkgstore.datahub.io/core/country-codes/country-codes_csv/data/c4df23af89a9386f92cbddcd54bc9852/country-codes_csv.csv'
