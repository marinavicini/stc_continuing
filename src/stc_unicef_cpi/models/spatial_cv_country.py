import cartopy.io.shapereader as shpreader
from pathlib import Path

import geopandas as gpd
import h3.api.numpy_int as h3
import re
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import numpy as np
import pycountry
import seaborn as sns
import swifter
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
from shapely.geometry import Point
from sklearn import clone, set_config
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score  # , log_loss
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from tqdm.auto import tqdm

from stc_unicef_cpi.data.cv_loaders import HexSpatialKFold, StratifiedIntervalKFold
from stc_unicef_cpi.features.build_features import boruta_shap_ftr_select
from stc_unicef_cpi.utils.mlflow_utils import fetch_logged_data
from stc_unicef_cpi.utils.scoring import mae

import stc_unicef_cpi.utils.constants as c
import stc_unicef_cpi.utils.clean_text as ct
import stc_unicef_cpi.utils.model_utils as mu


outputs = ['deprived_sev_mean_neigh', '2_or_more_prev_neigh', '3_or_more_prev_neigh', '4_or_more_prev_neigh',
        'housing_prev_neigh', 'water_prev_neigh', 'sanitation_prev_neigh', 
        'sumpoor_prev_neigh']
    #    'nutrition_prev_neigh', 'health_prev_neigh',
    #    'education_prev_neigh'

np.random.seed(seed=42)


################################################

country_code = 'COM'
cv_type = 'spatial'
nfolds = 5
test_size = 0.2
time_budget = 10
target_transform = None
standardise = 'robust'
impute = 'knn'
copy_to_nbrs = True
eval_split_type = 'normal'

################################################

# DATA_DIRECTORY = '/mnt/c/Users/vicin/Desktop/DSSG/Project/stc_continuing'
# mlflow_path = 'file:///mnt/c/Users/vicin/Desktop/DSSG/Project/models/mlruns'
mlflow_path = '/mnt/c/Users/vicin/Desktop/DSSG/Project/models/mlruns'

read_path = '/mnt/c/Users/vicin/Desktop/DSSG/Project/stc_continuing/data'

hexes_dhs = pd.read_csv(read_path + '/processed/20221102_hexes_dhs.csv', dtype={'hex_code':int})
hexes_dhs.shape

dhs_countries_code = list(hexes_dhs['country_code'].unique())
print(len(dhs_countries_code))
# for country_code in ['BEN']: 

# select country
df = mu.get_data_country(hexes_dhs, country_code, col='deprived_sev_count_neigh')

# drop country code
c.features.remove('country_code')
X, Y = df[c.features], df[outputs]
XY = df
print(XY.shape)


X_train, X_test, Y_all_train, Y_all_test = mu.select_eval_split_type(X, Y, eval_split_type='normal', test_size=test_size)
print(X_train.shape)
print(X_test.shape)

kfold, spatial_groups = mu.select_cv_type(cv_type=cv_type, nfolds=nfolds, XY=XY, X_train=X_train)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
Y_all_train.reset_index(drop=True, inplace=True)
Y_all_test.reset_index(drop=True, inplace=True)

# Y_all_train, Y_all_test = mu.select_target_transform(Y_all_train, Y_all_test, target_transform='none')


save = {}

for dim in outputs:
    Y_train = Y_all_train[dim]
    Y_test = Y_all_test[dim]

    mlflow.set_tracking_uri(f'file://{mlflow_path}') #MLFLOW_DIR)
    client = mlflow.tracking.MlflowClient()

    experiment_id = mu.call_experiment(client, 'spatialcv', country_code, dim)
    # print(experiment_id)


    for mod_type in ['lgbm']:
        
        num_imputer = mu.select_impute(impute=impute)
        col_tf = mu.col_transform(standardise=standardise, impute=impute)

        # model

        model = AutoML()
        automl_settings = {
            # CHANGE BUDGET TO 300 TO 30
                    "time_budget": time_budget,  # total running time in seconds for each target
                    "metric": "mse",  # primary metrics for regression can be chosen from: ['mae','mse','r2']
                    "task": "regression",  # task type
                    "estimator_list" : [mod_type],
                    # "n_jobs": args.ncores, WHAT IS IT
                    "log_file_name": "automl.log",  # flaml log file
                    "seed": 42,  # random seed
                    "eval_method": "cv",
                    "split_type": kfold,
                    "verbose":1,
                    "groups": spatial_groups if cv_type == "spatial" else None,
                }
        pipeline_settings = {
                    f"model__{key}": value for key, value in automl_settings.items()
                }

        pipeline = Pipeline([("impute", col_tf), ("model", model)])  

        pipeline.fit(X_train, Y_train, **pipeline_settings)
        automl = pipeline.steps[1][1]


        #### scores and save

        Y_pred = pipeline.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)

        save[f'{dim}_{mod_type}'] = [r2]
        
        with mlflow.start_run(experiment_id=experiment_id) as run: ########
            mlflow.set_tags({
                "country_code" : country_code,
                "target" : dim,
                "cv_type": cv_type,
                "eval_split_type": eval_split_type,
                "imputation": impute,
                "standardisation": standardise,
                "target_transform": target_transform,
                # "interpretable": args.interpretable,
                # "universal": args.universal_data_only,
                "copy_to_nbrs": copy_to_nbrs,
                "nfolds" : nfolds,
                "test_size" : test_size,
                "time_budget" : time_budget
                # "model_type": automl.best_estimator #############
                }
            )
        
            # metrics
            mu.mlflow_track_metrics(Y_pred, Y_test)
            mu.mlflow_track_automl(automl)
            mu.mlflow_plot(country_code, dim, Y_pred, Y_test) ##############################

        if (mod_type == 'lgbm') and (dim == 'deprived_sev_mean_neigh'):
            for cc in dhs_countries_code:
                if cc != country_code:
                    df_cc = mu.get_data_country(hexes_dhs, cc)
                    y_cc_pred = pipeline.predict(df_cc[c.features]) 
                    r2_cc = r2_score(df_cc[dim], y_cc_pred)
                    save[cc] = [r2_cc]
        

print(save)

quit()


        

for mod_type in ['lgbm']: #['xgboost', 'lgbm', 'rf', 'extra_tree']:

    # automl = AutoML()
    pipeline_settings['model__estimator_list'] = [mod_type]
    
    pipeline.fit(X_train, Y_train, **pipeline_settings)
    
    automl = pipeline.steps[1][1]

    Y_pred = pipeline.predict(X_test)
    r2_new = r2_score(Y_test, Y_pred)

    print(automl.model)
    print(r2_new)

    # SAVE
    save[mod_type] = [r2_new]

    if r2_new > r2:
        best_model = automl.model.model 
        r2 = r2_new

    with mlflow.start_run(experiment_id=experiment_id) as run: ########
        # Log with Mlflow
        # mu.mlflow_track_tags(country_code = country_code, 
        #                     dim = dim, 
        #                     cv_type = cv_type, 
        #                     eval_split_type = eval_split_type,
        #                     impute = impute, 
        #                     standardise = standardise, 
        #                     target_transform = target_transform, 
        #                     copy_to_nbrs = copy_to_nbrs, 
        #                     nfolds = nfolds, 
        #                     test_size = test_size, 
        #                     time_budget = time_budget)
        mlflow.set_tags({
            "country_code" : country_code,
            "target" : dim,
            "cv_type": cv_type,
            "eval_split_type": eval_split_type,
            "imputation": impute,
            "standardisation": standardise,
            "target_transform": target_transform,
            # "interpretable": args.interpretable,
            # "universal": args.universal_data_only,
            "copy_to_nbrs": copy_to_nbrs,
            "nfolds" : nfolds,
            "test_size" : test_size,
            "time_budget" : time_budget
            # "model_type": automl.best_estimator #############
            }
        )
        
        # metrics
        mu.mlflow_track_metrics(Y_pred, Y_test)
        mu.mlflow_track_automl(automl)
        mu.mlflow_plot(country_code, dim, Y_pred, Y_test) ##############################



pipeline_best = Pipeline([("impute", col_tf), ("model", best_model)])
pipeline_best.fit(X, Y[dim])

# for cc in dhs_countries_code:
#     if cc != country_code:
#         df_cc = mu.get_data_country(hexes_dhs, cc)
#         y_cc_pred = pipeline_best.predict(df_cc[c.features]) 
#         r2_cc = r2_score(df_cc[dim], y_cc_pred)
#         save[cc] = [r2_cc]



for dim in outputs:
    if dim=='deprived_sev_mean_neigh':
        continue

    Y_train = Y_all_train[dim]
    Y_test = Y_all_test[dim]
    print(dim)
    dim_clean = ct.clean_name_dim(dim)

    experiment_id = mu.call_experiment(client, 'spatialcv', country_code, dim_clean)

    automl_settings['estimator_list'] = ["xgboost", "lgbm", "rf", "extra_tree"]
    automl_settings['time_budget'] = time_budget

    automl = AutoML()
    automl.fit(X_train, Y_train, **automl_settings)

    Y_pred = automl.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    save[dim] = [r2]
    save[f'{dim}_model'] = [automl.best_estimator]
    with mlflow.start_run(experiment_id=experiment_id) as run: 
        # Log with Mlflow
        mlflow.set_tags({
            "country_code" : country_code,
            "target" : dim,
            "cv_type": cv_type,
            "eval_split_type": eval_split_type,
            "imputation": impute,
            "standardisation": standardise,
            "target_transform": target_transform,
            # "interpretable": args.interpretable,
            # "universal": args.universal_data_only,
            "copy_to_nbrs": copy_to_nbrs,
            "nfolds" : nfolds,
            "test_size" : test_size,
            "time_budget" : time_budget
            # "model_type": automl.best_estimator 
            }
        )

        # metrics
        # mlflow.log_metric(key="r2_score", value=r2)
        # mse_val = sklearn_metric_loss_score("mse", Y_pred, Y_test) 
        # mlflow.log_metric(key="mse", value=mse_val) 
        # mae_val = sklearn_metric_loss_score("mae", Y_pred, Y_test) 
        # mlflow.log_metric(key="mae", value=mae_val) 
        
        mu.mlflow_track_metrics(Y_pred, Y_test)
        mu.mlflow_track_automl(automl)
        mu.mlflow_plot(country_code, dim, Y_pred, Y_test) ##############################



print(save)

data = pd.DataFrame.from_dict(save)
data.to_csv(f'/mnt/c/Users/vicin/Desktop/DSSG/Project/stc_continuing/data/processed/final/{country_code}_results.csv', index = False)

# SAVE
# fit the model on all the data xtrain + xtest 
# and then predict on all the remaining countries


