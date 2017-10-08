#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
reload(sys)
sys.setdefaultencoding('utf8') # problem with encoding

import matplotlib
matplotlib.use("Qt4Agg") # enable plt.show() to display
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, tzinfo
from dateutil import tz

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV)

from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest, chi2

# Personals files
import plot
import handle_datas


# Constants for reproductability
SEED = 314
TEST_SIZE = 0.2
MAX_EVALS = 100

##############################################
# Global variables :
script_path = os.path.abspath(sys.argv[0])
working_dir = os.path.dirname(script_path)

data_dir = working_dir + "/data/"
data_reformated_dir = working_dir + "/reformated_data/"
results_dir = working_dir + "/results/"


def set_column_sequence(dataframe, seq, front=True):
    '''Takes a dataframe and a subsequence of its columns,
       returns dataframe with seq as first columns if "front" is True,
       and seq as last columns if "front" is False.
    '''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            if front: #we want "seq" to be in the front
                #so append current column to the end of the list
                cols.append(x)
            else:
                #we want "seq" to be last, so insert this
                #column in the front of the new column list
                #"cols" we are building:
                cols.insert(0, x)
    return dataframe[cols]


def get_bounds_datetime(df1, df2, *dfs):
    # Youngest admissible to avoid missing values :
    dt_min = max(min(df1['Date']), min(df2['Date']))
    # Oldest admissible to avoid missing values :
    dt_max = min(max(df1['Date']), max(df2['Date']))

    for df in dfs:
        dt_min = max(dt_min, min(df['Date']))
        dt_max = min(dt_max, max(df['Date']))

    return dt_min, dt_max


def drop_outof_dt_bounds(dt_min, dt_max, df):
    df_tmp = df[(df['Date'] >= dt_min) & (df['Date'] <= dt_max)]
    return df_tmp


def get_df_turbines(lst_turb):
    df_turb = pd.DataFrame()
    for num_turb in lst_turb:
        fname_in_turb = data_reformated_dir + 'turb_' + str(num_turb) + '.csv'
        df_turb = df_turb.append(pd.read_csv(fname_in_turb, sep=';'))

    df_turb['Date'] = pd.to_datetime(df_turb['Date'], format="%Y-%m-%d %H:%M:%S")

    # Convert hourly
    # !TO DO : better the model by splitting according to Fonctionnement per
    # minute
    # Right now, only keeping dt.minute == 0 :
    df_turb = df_turb[(df_turb['Date'].dt.minute == 0)
                      & (df_turb['Fonctionnement'] == 1)]
    df_turb.drop('Fonctionnement', axis=1, inplace=True)
    df_turb.drop('Production', axis=1, inplace=True)
    df_turb.sort_values('Date', ascending=1, inplace=True)
    return df_turb


def get_df_turb_2017():
    df_turb_2017 = handle_datas.create_df_park_data(
        list_num_park=[1, 2, 3], list_date_park=['2017'])
    df_turb_2017['Date'] = pd.to_datetime(
        df_turb_2017['Date'], format="%d/%m/%Y %H:%M")

    df_turb_2017 = df_turb_2017[
        (df_turb_2017['Date'].dt.minute == 0) &
        (df_turb_2017['Fonctionnement'] == 1)]
    df_turb_2017 = df_turb_2017[['Date', 'Eolienne']]
    df_turb_2017.sort_values(by='Date', ascending=True, inplace=True)
    return df_turb_2017


def get_df_weather(lst_grid, year=None):
    if not all(num_grid > 0 for num_grid in lst_grid):
        raise Exception('Error in get_df_weather_red : grid_id are only positives')
    elif len(lst_grid) == 0:
        return pd.DataFrame()

    df_weather = pd.DataFrame()
    for num_grid in lst_grid:
        if year == '2017':
            fname_in_grid = data_reformated_dir + 'PrevMeteo_Grille' +\
                            str(num_grid) + '_2017' + '.csv'
        else:
            fname_in_grid = data_reformated_dir + 'PrevMeteo_Grille' +\
                            str(num_grid) + '.csv'

        df_tmp = pd.read_csv(fname_in_grid, sep=';')
        df_tmp['grid_id'] = pd.Series(num_grid, index=df_tmp.index)

        df_weather = df_weather.append(df_tmp)

    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format="%Y-%m-%d %H:%M:%S")

    df_weather = set_column_sequence(df_weather, ['Date', 'fc_hor', 'grid_id'])
    df_weather.sort_values(by=['Date', 'fc_hor', 'grid_id'],
                           ascending=[True, True, True],
                           inplace=True)
    return df_weather


def filter_fc_hor(df, lst_col_to_group=['Date', 'grid_id'], lst_da=[2]):
    # Rem :
    # The aim of this challenge is to predict the Production of tomorrow.
    # So only forecast (fc_hor) between 25h and 47h can be used to predict:
    df_tmp = pd.DataFrame()
    for da_num in lst_da:
        if da_num == 1:
            df_tmp = df_tmp.append( df[(df['fc_hor'] <= 23)] )
        if da_num == 2:
            df_tmp = df_tmp.append( df[(df['fc_hor'] >= 24) & (df['fc_hor'] <= 47)] )
        if da_num == 3:
            df_tmp = df_tmp.append( df[(df['fc_hor'] >= 48 )] )

    # Append doesn't identify common Date & grid_id, need to groupby :
    if len(lst_da) > 1:
        df_tmp = df_tmp.groupby(lst_col_to_group).mean().reset_index()

    df_tmp.sort_values(by=lst_col_to_group,
                           ascending=len(lst_col_to_group)*[True],
                           inplace=True)
    return df_tmp


def merge_df_turb_weather(df_turb, df_weather, lst_da=[2]):
    df_weather = filter_fc_hor(df_weather, lst_da=lst_da)

    dt_min, dt_max = get_bounds_datetime(df_turb, df_weather)

    df_turb = drop_outof_dt_bounds(dt_min, dt_max, df_turb)
    df_weather = drop_outof_dt_bounds(dt_min, dt_max, df_weather)

    df = pd.merge(df_turb, df_weather, on='Date', how='left')

    # NA values exists because some datetimes doesn't have a fc_hor in the
    # interval needed
    df.dropna(inplace=True)
    df.sort_values(by='Date', ascending=True, inplace=True)
    return df


lst_col_model_red = ['RS', 'CAPE', 'SP', 'CP',
                 'BLD', 'SSHF', 'SLHF', 'MSL', 'BLH', 'TCC', 'U10', 'V10', 'T2',
                 'D2', 'SSRD', 'STRD', 'SSR', 'STR', 'TSR', 'LCC', 'MCC', 'HCC',
                 'TSRC', 'SSRC', 'STRC', 'TP', 'FA', 'U100', 'V100', 'vit_100',
                 'vit_10', 'dir_100', 'dir_10']


# lst_col_model = ['SP', 'CP',
#                  'BLD', 'SSHF', 'SLHF', 'MSL', 'BLH', 'TCC', 'U10', 'V10', 'T2',
#                  'D2', 'STRD', 'SSR', 'STR', 'TSR', 'LCC', 'MCC', 'HCC',
#                  'STRC', 'TP', 'FA', 'U100', 'V100', 'vit_100',
#                  'vit_10', 'dir_100', 'dir_10']

dt_start_pred = datetime(2017, 1, 1, 0, 0)
dt_end_pred = datetime(2017, 4, 14, 23, 0)

def param_xgboost_to_pipe(basename='regressor', **param_xgboost):
    if not bool(param_xgboost):  # empty dict are evaluated to False in python
        return {}
    else:
        param_grid = {}
        for var_name in param_xgboost:
            param_grid[basename+'__'+var_name] = param_xgboost[var_name]
        return param_grid


class model_ml:
    def __init__(self, lst_turb=[1], lst_grid=[8, 9], lst_da_train=[2],
                 lst_col_model=lst_col_model_red,
                 col_target='Production_mean_hour',
                 submit_mode=False):

        # Storing config :
        self.submit_mode = submit_mode

        if submit_mode:
            print('---> Submit mode activated ! <---')
            self.lst_turb = range(1, 12)
        else:
            print('---> Just testing mode <---')
            self.lst_turb = lst_turb
            self.scores = pd.DataFrame(columns=['lst_turb', 'lst_grid',
                                                'Model', 'Params', 'MAE'])

        self.lst_grid = lst_grid

        self.lst_da_train = lst_da_train
        self.lst_da_test = [2]  # Product constraints

        self.lst_col_model = lst_col_model
        self.col_target = col_target


    def get_datas_2017(self):
        # dt_start_pred < dt < dt_end_pred :
        print('Reading 2017 weather csv ...')
        df_weather_2017 = get_df_weather(lst_grid=self.lst_grid,
                                        year='2017')
        df_weather_2017 = drop_outof_dt_bounds(
            dt_start_pred, dt_end_pred, df_weather_2017)

        print('Reading 2017 parks csv ...')
        df_turb_2017 = get_df_turb_2017()
        df_turb_2017 = drop_outof_dt_bounds(
            dt_start_pred, dt_end_pred, df_turb_2017)

        # Production constraints : can only predict with forecast 1da
        # meaning fc_hor >= 24 & fc_hor <= 47   cf. self.lst_da_test
        self.df_2017 = merge_df_turb_weather(
            df_turb=df_turb_2017, df_weather=df_weather_2017,
            lst_da=self.lst_da_test)


    def get_datas(self):
        print('Reading 2015 & 2016 parks csv ...')
        df_turb = get_df_turbines(lst_turb=self.lst_turb)
        print('Reading 2015 & 2016 weather csv ...')
        df_weather = get_df_weather(lst_grid=self.lst_grid)

        self.df = merge_df_turb_weather(
            df_turb=df_turb, df_weather=df_weather,
            lst_da=[1,2,3])

        if self.submit_mode:
            self.get_datas_2017()
            self.df = pd.concat([self.df, self.df_2017])

        # Adding turb_id column :
        self.df['turb_id'] = self.df['Eolienne'].apply(
            lambda x: int(x[4:])) # Filter 'Turb' from 'Turbx[x]'
        self.df = set_column_sequence(
            self.df,
            ['Date', 'Production_mean_hour', 'fc_hor', 'grid_id', 'turb_id'])


    def split_train_test(self, conveniance_fetch=True):

        if self.submit_mode:
            # Train set :
            dt_start_train = min(self.df['Date'])
            self.df_train = drop_outof_dt_bounds(
                dt_start_train, dt_start_pred - timedelta(hours=1), self.df)
            self.df_train = filter_fc_hor(
                self.df_train,
                lst_col_to_group=['Date', 'grid_id', 'turb_id'],
                lst_da=self.lst_da_train)

            # Test set :
            self.df_test = drop_outof_dt_bounds(dt_start_pred, dt_end_pred,
                                                self.df)
            self.df_test = filter_fc_hor(
                self.df_test,
                lst_col_to_group=['Date', 'grid_id', 'turb_id'],
                lst_da=self.lst_da_test)

        else:
            # Train - Test split :
            self.df_train, self.df_test = train_test_split(
                self.df, test_size=TEST_SIZE, random_state=SEED)
            # Conveniance fetch :
            if conveniance_fetch:
                self.X = self.df[self.lst_col_model]
                self.y = self.df[self.col_target]
                self.X_train = self.df_train[self.lst_col_model]
                self.y_train = self.df_train[self.col_target]
                self.X_test = self.df_test[self.lst_col_model]
                self.y_test = self.df_test[self.col_target]


    def filter_outliers(self, zscore_max_abs=3):
        from scipy.stats import zscore
        self.df_train[self.lst_col_model] = self.df_train[self.lst_col_model]\
            [(np.abs(zscore(self.df_train[self.lst_col_model])) < zscore_max_abs)
             .all(axis=1)]
        self.df_train.dropna(inplace=True)


    def compute(self, modeltype=XGBRegressor, early_stopping_rounds=None, **kwargs):
        print('-------------------------------------------')
        if kwargs != {}:
            print(kwargs)

        self.model = modeltype(**kwargs)

        if early_stopping_rounds == None:
            print('Training ' + str(modeltype) + ' model ...')
            self.model.fit(self.df_train[self.lst_col_model],
                           self.df_train[self.col_target],
                           verbose=True,
                           )
        else:
            print('Training ' + str(modeltype) +
                  ' model with early_stopping_rounds = ' +
                  str(early_stopping_rounds) + '...')

            self.model.fit(self.df_train[self.lst_col_model],
                           self.df_train[self.col_target],
                           early_stopping_rounds=early_stopping_rounds,
                           eval_set=[(self.df_test[self.lst_col_model],
                                      self.df_test[self.col_target])],
                           verbose=True,
                           )

        print('Predicting ' + str(modeltype) + ' model ...')
        self.df_test['pred'] = self.model.predict(
            self.df_test[self.lst_col_model])

        if not self.submit_mode:
            mae = mean_absolute_error(self.df_test['pred']
                                      , self.df_test[self.col_target])
            self.scores.loc[self.scores.shape[0]] =\
                [self.lst_turb, self.lst_grid, str(modeltype), kwargs, mae]
            return mae


    def compute_submit_nturb_models(self, modeltype, **kwargs):
        df_test = pd.DataFrame()
        if kwargs != {}:
            print(kwargs)

        for nturb in self.lst_turb:
            print('-------------------------------------------')
            df_train_tmp = self.df_train[self.df_train['Eolienne']
                                         == 'Turb' + str(nturb)].copy()
            df_test_tmp = self.df_test[self.df_test['Eolienne']
                                       == 'Turb' + str(nturb)].copy()

            print('Processing turbine ' + str(nturb))
            model = modeltype(**kwargs)
            model.fit(df_train_tmp[self.lst_col_model],
                      df_train_tmp[self.col_target])
            df_test_tmp['pred'] = model.predict(df_test_tmp[self.lst_col_model])
            df_test = df_test.append(df_test_tmp)

        self.df_test = df_test

        if not self.submit_mode:
            mae = mean_absolute_error(self.df_test['pred']
                                      , self.df_test[self.col_target])
            self.scores.loc[self.scores.shape[0]] =\
                [self.lst_turb, self.lst_grid, str(modeltype), kwargs, mae]
            return mae


    def write_submit_csv(self):
        now = datetime.now()
        fname_out = results_dir + 'lcv_submit_GJ_' + str(now.day) +\
            '_' + str(now.hour) + '_' + str(now.minute) + '.csv'
        # Computing the mean production predicted for each hour according to
        # grid id :
        self.df_test['Eolienne'] = self.df_test['turb_id'].apply(
            lambda x: 'Turb' + str(x))
        df = self.df_test[['Date', 'Eolienne', 'pred']]
        df = df.groupby(['Date', 'Eolienne']).mean().reset_index()
        df.to_csv(fname_out, sep=';', index=False)


    def train_grid_search(self, param_xgboost, n_jobs=3,
                          model_ml_type=XGBRegressor):

        pipe = Pipeline(steps=[
            # ('reduce_dim', decomposition.PCA()),
            ('regressor', model_ml_type())
        ])

        # N_FEATURES_OPTIONS = [24, 27, 29, 31, 33]
        # param_grid = [
        #     {
        #         # 'reduce_dim': [decomposition.PCA()],
        #         # 'reduce_dim__n_components': N_FEATURES_OPTIONS,
        #         'regressor__max_depth': [4, 6, 10, 15],
        #         'regressor__n_estimators': [10, 50, 100, 500],
        #         'regressor__learning_rate': [0.01, 0.025, 0.05, 0.1],
        #         'regressor__gamma': [0.05, 0.5, 0.9, 1.]
        #     },
        # ]

        param_grid = param_xgboost_to_pipe(**param_xgboost)
        fit_params = {'regressor__early_stopping_rounds': 20,
                      'regressor__eval_set': [
                          (self.df_test[self.lst_col_model],
                           self.df_test[self.col_target])],
                     }

        grid = GridSearchCV(pipe, cv=5, verbose=2, n_jobs=n_jobs,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',  #mean_absolute_error
                            fit_params=fit_params,
                            )
        grid.fit(self.df_train[self.lst_col_model],
                 self.df_train[self.col_target])

        self.grid = grid


    def feature_engineering(self):
        # Add Eolienne, fc_hor and grid_id to the model :
        self.lst_col_model += ['fc_hor', 'grid_id', 'turb_id']

        # Let's take into account Datetime
        self.df['month'] = self.df['Date'].apply(lambda x: x.month)
        self.df['day'] = self.df['Date'].apply(lambda x: x.day)
        self.df['hour'] = self.df['Date'].apply(lambda x: x.hour)
        self.lst_col_model += ['month', 'day', 'hour']

        # # Combine fc_hor and hour
        # self.df['delta_hour_fc_hor'] = self.df['fc_hor'] - self.df['hour']
        # self.lst_col_model += ['delta_hour_fc_hor']
        # self.lst_col_model.remove('hour')
        # self.lst_col_model.remove('fc_hor')

        # Doing a sum for the booleans cloud cover :
        self.df['cloud_cover'] = self.df[['LCC', 'MCC', 'HCC']].sum(axis=1)
        self.lst_col_model = [x for x in self.lst_col_model if x not in
                            ['LCC', 'MCC', 'HCC']]
        self.lst_col_model += ['cloud_cover']

        # Delete some not important variables :
        self.lst_col_model = [x for x in self.lst_col_model if x not in
                            ['SSRD', 'TSR', 'TSRC']]



# Some checks :
if not os.path.isdir(data_dir):
    raise Exception(data_dir + " doesn't exist.")

if not os.path.isdir(data_reformated_dir):
    os.makedirs(data_reformated_dir)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

naive_cv_parameters = {'max_depth':[4, 6, 8, 10],
                       'n_estimators': [10, 15, 20, 25],
                       'learning_rate': [0.2, 0.4, 0.6, 0.8],
                       'gamma': [0.2, 0.4, 0.6, 0.8]
                       }

test_cv_parameters = {'max_depth':[4],
                      'n_estimators': [10, 15],
                      'learning_rate': [0.2],
                       }

expert_cv_parameters = {'max_depth':[4, 6, 10, 15],
                        'n_estimators': [10, 50, 100, 500],
                        'learning_rate': [0.01, 0.025, 0.05, 0.1],
                        'gamma': [0.05, 0.5, 0.9, 1.], # Minimum loss reduction required to make a further partition on a leaf node of the tree
                        }

optimal_params = {'max_depth': 8, 'n_estimators': 300, 'learning_rate': 0.05}


# m = model_ml(lst_turb=range(1, 12), lst_grid=range(1,17))
# m = model_ml(lst_turb=[1], lst_grid=[6,7,8,11], lst_da_train=[2])

m = model_ml(lst_grid=range(1,17), submit_mode=True, lst_da_train=[1,2])
m.get_datas()
m.feature_engineering()
m.split_train_test()

