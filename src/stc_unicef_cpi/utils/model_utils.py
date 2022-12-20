import pandas as pd
import numpy as np

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import h3.api.numpy_int as h3
import re
import joblib
import matplotlib.pyplot as plt
import mlflow
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



def get_data_country(data, country_code, col = 'deprived_sev_count_neigh'):
    '''select data of a country with threshold of 30'''
    df = data[data['country_code'] == country_code]
    # hexagons above 30 threshold
    df = df[df[col].isna()==False]
    df = df[df[col]>30]
    return df


def select_standardise(standardise='none'):
    if standardise == "none":
        num_stand = None
    elif standardise == "standard":
        num_stand = StandardScaler()
    elif standardise == "minmax":
        num_stand = MinMaxScaler()
    elif standardise == "robust":
        num_stand = RobustScaler()
    
    return num_stand


def select_impute(impute='none'):
    # imputer setup
    # choices = ["none", "mean", "median", "knn", "linear", "rf"]
    add_indicator = True
    if impute == "none":
        num_imputer = None
    elif impute == "mean":
        # default strategy is mean
        num_imputer = SimpleImputer(add_indicator=add_indicator)
    elif impute == "median":
        num_imputer = SimpleImputer(strategy="median", add_indicator=add_indicator)
    elif impute == "knn":
        num_imputer = KNNImputer(n_neighbors=5, add_indicator=add_indicator)
    elif impute == "linear":
        # default estimator is BayesianRidge
        num_imputer = IterativeImputer(
            max_iter=10, random_state=42, add_indicator=add_indicator
        )
    elif impute == "rf":
        num_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=20, min_samples_split=5, min_samples_leaf=3
            ),
            max_iter=10,
            random_state=42,
            add_indicator=add_indicator,
        )

    return num_imputer


def col_transform(standardise='none', impute='none'):
    num_imputer = select_impute(impute=impute)
    num_stand = select_standardise(standardise=standardise)
    num_tf = Pipeline(steps=[("imputer", num_imputer), ("standardiser", num_stand)])

    # as only commuting_zn is cat, just use constant imputation for this (only 5 missing records)
    cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")

    cat_tf = Pipeline(steps=[("imputer", cat_imputer),
            #  ("encoder", cat_enc)
        ])
    col_tf = make_column_transformer(
        (num_tf, make_column_selector(dtype_include=np.number)),
        (cat_tf, make_column_selector(dtype_exclude=np.number)),
    )
    return col_tf

# def get_categorical(X):
#     categorical_features = X.select_dtypes(exclude=[np.number]).columns
#     X[categorical_features] = X[categorical_features].astype("category")


def select_target_transform(Y_train, Y_test, target_transform='none'):
    # target transforms
    # choices = ["none", "log", "power"]
    if target_transform != "none":
        if target_transform == "log":
            Y_train = np.log(Y_train)
            Y_test = np.log(Y_test)
        elif target_transform == "power":
            power_tf = PowerTransformer().fit(Y_train)
            Y_train = power_tf.transform(Y_train)
            Y_test = power_tf.transform(Y_test)
        else:
            raise ValueError("Invalid target transform")

    return Y_train, Y_test


def select_eval_split_type(X, Y, eval_split_type='normal', test_size=0.2):
    # generate train / test split
    # choices = ["normal", "stratified", "spatial"]

    if eval_split_type == "normal":
        # TODO: rerun without fixing random state for general splits
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )
    elif eval_split_type == "stratified":
        if len(Y.shape) == 1:
            train_idxs, test_idxs = next(
                StratifiedIntervalKFold(
                    n_splits=int(1 / test_size), random_state=42
                ).split(X, Y)
            )
            X_train, Y_train = X[train_idxs], Y[train_idxs]
            X_test, Y_test = X[test_idxs], Y[test_idxs]
        else:
            # must generate split for each column separately
            strat_X_train = {}
            strat_Y_train = {}
            strat_X_test = {}
            strat_Y_test = {}
            for col_idx in range(Y.shape[1]):
                train_idxs, test_idxs = next(
                    StratifiedIntervalKFold(
                        n_splits=int(1 / test_size), random_state=42
                    ).split(X, Y.iloc[:, col_idx])
                )
                X_train, y_train = X[train_idxs], Y.iloc[:, col_idx][train_idxs]
                X_test, y_test = X[test_idxs], Y.iloc[:, col_idx][test_idxs]
                strat_X_train[col_idx] = X_train
                strat_Y_train[col_idx] = y_train
                strat_X_test[col_idx] = X_test
                strat_Y_test[col_idx] = y_test
    elif eval_split_type == "spatial":
        train_idxs, test_idxs = next(
            HexSpatialKFold(n_splits=int(1 / test_size), random_state=42).split(
                X, Y
            )
        )
        X_train, Y_train = X[train_idxs], Y[train_idxs]
        X_test, Y_test = X[test_idxs], Y[test_idxs]

    return X_train, X_test, Y_train, Y_test


def select_cv_type(cv_type='normal', nfolds=5, XY=None, X_train=None):
    ''' Specify KFold strategy
    choices = ["normal", "stratified", "spatial"]
    If 'spatial' then need to specify XY and X_train
    '''
    spatial_groups=None

    if cv_type == "normal":
        kfold = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    elif cv_type == "stratified":
        kfold = StratifiedIntervalKFold(n_splits=nfolds, shuffle=True, n_cuts=5, random_state=42)
    elif cv_type == "spatial":
        # print(X.iloc[:,-1].head())
        kfold = GroupKFold(n_splits= nfolds)
        spatial_groups = HexSpatialKFold(n_splits=nfolds, random_state=42).get_spatial_groups(
            XY["hex_code"].loc[X_train.index]
        )
        try:
            assert len(spatial_groups) == len(X_train)
        except AssertionError:
            print(spatial_groups.shape, X_train.shape)
    else:
        raise ValueError("Invalid CV type")
    
    return kfold, spatial_groups
   

def call_experiment(client, cv_type, country_code, target):
    '''check if experiment exists otherwise create it'''
    target_name = ct.clean_name_dim(target)

    try:
        # Create an experiment name, which must be unique and case sensitive
        experiment_id = client.create_experiment(
            f"{cv_type}-{country_code}-{target}",
            tags={"cv_type": cv_type, "country_code": country_code, "target":target_name},
        )
        # experiment = client.get_experiment(experiment_id)
    except:
        assert (
            f"{cv_type}-{country_code}-{target}"
            in [exp.name for exp in client.list_experiments()]
        )
        experiment_id = f"{cv_type}-{country_code}-{target}"
        experiment_id = [
            exp.experiment_id
            for exp in client.list_experiments()
            if exp.name
            == f"{cv_type}-{country_code}-{target}"
        ][0]
    return experiment_id


def mlflow_track_automl(automl):
    mlflow.log_param(key="best_model", value=automl.best_estimator)
    mlflow.log_param(key="best_config", value=automl.best_config)
    mlflow.log_params(automl.best_config) #### WHAT TO USE???

    # metrics
    mlflow.log_metric(key="pred_time", value=automl.best_result['pred_time']) 
    mlflow.log_metric(key="validation_loss", value=automl.best_result['val_loss']) 
    mlflow.log_metric(key="wall_clock_time", value=automl.best_result['wall_clock_time']) 
    mlflow.log_metric(key="training_iteration", value=automl.best_result['training_iteration']) 
    
    # model
    mlflow.sklearn.log_model(automl.model.model, "model") ###


def mlflow_track_tags(country_code, dim, cv_type, eval_split_type, impute, standardise, target_transform, copy_to_nbrs, nfolds, test_size, time_budget):
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

def mlflow_track_metrics(Y_pred, Y_test):
    '''track test metrics'''
    
    r2 = r2_score(Y_test, Y_pred)
    mlflow.log_metric(key="r2_score", value=r2)
    mse_val = sklearn_metric_loss_score("mse", Y_pred, Y_test) 
    mlflow.log_metric(key="mse", value=mse_val) 
    mae_val = sklearn_metric_loss_score("mae", Y_pred, Y_test) 
    mlflow.log_metric(key="mae", value=mae_val) 



def mlflow_plot(country_code, dim, Y_pred, Y_test):
    fig, ax = plt.subplots()

    dim = ct.clean_name_dim(dim)
    # all dim except depth are between 0 and 1
    if dim == 'sumpoor':
        mi, ma = 0, 3.5
    else:
        mi, ma = 0, 1

    ax.plot(Y_pred, Y_test, '.')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xlim([mi, ma])
    plt.ylim([mi, ma])

    mlflow.log_figure(fig, f'plot_{country_code}_{dim}.png')



def mlflow_track_feature_imp(pipeline, X, Y):

    assert Y.shape[1] == 1

    ftr_names = pipeline[:-1].get_feature_names_out()
    ftr_names = [
                    name.split("__")[1] if "__" in name else name for name in ftr_names
                ]

    X_tf = pd.DataFrame(
                    pipeline[:-1].transform(X), columns=ftr_names, index=X.index
                )

    categorical_features = X_tf.select_dtypes(exclude=[np.number]).columns
    try:
        X_tf[categorical_features] = pd.concat(
            [
                X_tf[cat_col].astype("category").cat.codes
                for cat_col in categorical_features
            ],
            axis=1,
        ).astype("category")
    except:
        pass

    automl = pipeline.steps[1][1]
    ftr_subset = boruta_shap_ftr_select(
        X_tf,
        Y,
        base_model=clone(automl.model.estimator),
        plot=True,
        n_trials=100,
        sample=False,
        train_or_test="test",
        normalize=True,
        verbose=True,
        incl_tentative=True,
    )


def automl_feat_importance(pipeline, thres = 0):
    thres = 0

    ftr_names = pipeline[:-1].get_feature_names_out()
    ftr_names = [name.split("__")[1] if "__" in name else name for name in ftr_names]

    automl = pipeline.steps[1][1]

    not_zeros = automl.feature_importances_ > thres
    feat_names = pd.Series(ftr_names)[list(not_zeros)] #automl.feature_names_in_
    feat_imp = pd.Series(automl.feature_importances_)[list(not_zeros)]

    fig, ax = plt.subplots()
    ax.barh(feat_names, feat_imp)
    return fig


def mlflow_track_automl_ft_imp(pipeline, country_code, dim, thres = 0):
    automl = pipeline.steps[1][1]

    mod = automl.best_estimator
    dim = ct.clean_name_dim(dim)
    fig = automl_feat_importance(pipeline, thres)
    mlflow.log_figure(fig, f'plot_{country_code}_{dim}_{mod}.png')

