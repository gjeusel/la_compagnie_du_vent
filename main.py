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

import plot
import handle_datas


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
    df = df[(df['Date'] >= dt_min) & (df['Date'] <= dt_max)].copy()
    return df


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


def get_df_weather_red(lst_grid):
    if not all(num_grid > 0 for num_grid in lst_grid):
        raise Exception('Error in get_df_weather_red : grid_id are only positives')
    elif len(lst_grid) == 0:
        return pd.DataFrame()

    df_weather = pd.DataFrame()
    for num_grid in lst_grid:
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


def get_df_weather_yellow(lst_grid):
    if not all(num_grid < 0 for num_grid in lst_grid):
        raise Exception('Error in get_df_weather_yellow : grid_id are only negatives')
    elif len(lst_grid) == 0:
        return pd.DataFrame()

    df_weather = pd.DataFrame()
    for num_grid in lst_grid:
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


def get_df_weather(lst_grid):
    df_red = get_df_weather_red([n for n in lst_grid if n > 0])
    df_yellow = get_df_weather_yellow([n for n in lst_grid if n < 0])

    if (df_red.empty):
        df = df_yellow
    elif (df_yellow.empty):
        df = df_red
    else:
        dt_min, dt_max = get_bounds_datetime(df_red, df_yellow)
        df_red = drop_outof_dt_bounds(dt_min, dt_max, df_red)
        df_yellow = drop_outof_dt_bounds(dt_min, dt_max, df_yellow)

        # GOT TO CHECK for same fc_hor
        df = pd.merge(df_red, df_yellow, on=['Date', 'fc_hor', 'grid_id']
                      , how='left')

    return df


def filter_weather_fc_hor(df, lst_da=[2]):
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

    if len(lst_da) > 1:
        df_tmp = df_tmp.groupby(['Date', 'grid_id']).mean().reset_index()

    # df_tmp.drop('fc_hor', axis=1, inplace=True)
    df_tmp.sort_values(by=['Date', 'grid_id'],
                           ascending=[True, True],
                           inplace=True)
    return df_tmp


def merge_df_turb_weather(df_turb, df_weather, lst_da=[2]):
    df_weather = filter_weather_fc_hor(df_weather, lst_da=lst_da)

    dt_min, dt_max = get_bounds_datetime(df_turb, df_weather)

    df_turb = drop_outof_dt_bounds(dt_min, dt_max, df_turb)
    df_weather = drop_outof_dt_bounds(dt_min, dt_max, df_weather)

    df = pd.merge(df_turb, df_weather, on='Date', how='left')

    # NA values exists because some datetimes doesn't have a fc_hor in the
    # interval needed
    df.dropna(inplace=True)
    df.sort_values(by='Date', ascending=True, inplace=True)
    return df


lst_col_model = ['RS', 'CAPE', 'SP', 'CP',
                 'BLD', 'SSHF', 'SLHF', 'MSL', 'BLH', 'TCC', 'U10', 'V10', 'T2',
                 'D2', 'SSRD', 'STRD', 'SSR', 'STR', 'TSR', 'LCC', 'MCC', 'HCC',
                 'TSRC', 'SSRC', 'STRC', 'TP', 'FA', 'U100', 'V100', 'vit_100',
                 'vit_10', 'dir_100', 'dir_10']

# lst_col_model = ['SP', 'CP',
#                  'BLD', 'SSHF', 'SLHF', 'MSL', 'BLH', 'TCC', 'U10', 'V10', 'T2',
#                  'D2', 'STRD', 'SSR', 'STR', 'TSR', 'LCC', 'MCC', 'HCC',
#                  'STRC', 'TP', 'FA', 'U100', 'V100', 'vit_100',
#                  'vit_10', 'dir_100', 'dir_10']

lst_col_prev = ['Date', 'Eolienne', 'pred']

dt_start_pred = datetime(2017, 1, 1, 0, 0)
dt_end_pred = datetime(2017, 4, 14, 23, 0)


class model_ml:
    def __init__(self, lst_turb=[1, 2], lst_grid=[8, 9],
                 lst_da_train=[2],
                 lst_col_model=lst_col_model,
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


    def get_datas(self):
        print('Reading 2015-2016 parks csv ...')
        df_turb = get_df_turbines(lst_turb=self.lst_turb)
        print('Reading 2015-2017 weather csv ...')
        df_weather = get_df_weather_red(lst_grid=self.lst_grid)

        if self.submit_mode:
            # Preparing test sample for submit, only values with produc
            # constraints and dt_start_pred < dt < dt_end_pred :
            print('Reading 2017 parks csv ...')
            df_turb_2017 = get_df_turb_2017()
            df_turb_2017 = drop_outof_dt_bounds(
                dt_start_pred, dt_end_pred, df_turb_2017)

            # Production constraints : can only predict with forecast 1da
            # meaning fc_hor >= 24 & fc_hor <= 47   cf. self.lst_da_test
            self.df_test = merge_df_turb_weather(
                df_turb=df_turb_2017, df_weather=df_weather,
                lst_da=self.lst_da_test)

            self.df_train = merge_df_turb_weather(
                df_turb=df_turb, df_weather=df_weather,
                lst_da=self.lst_da_train)

        else:
            from sklearn.model_selection import train_test_split
            df = merge_df_turb_weather(df_turb, df_weather,
                                       lst_da=self.lst_da_train)
            self.df_train, self.df_test = train_test_split(df, test_size=0.2)


    def filter_outliers(self, zscore_max_abs=3):
        from scipy.stats import zscore
        self.df_train[self.lst_col_model] = self.df_train[self.lst_col_model]\
            [(np.abs(zscore(self.df_train[self.lst_col_model])) < zscore_max_abs)
             .all(axis=1)]
        self.df_train.dropna(inplace=True)


    def compute(self, modeltype, **kwargs):
        print('-------------------------------------------')
        if kwargs != {}:
            print(kwargs)

        model = modeltype(**kwargs)

        print('Training ' + str(modeltype) + ' model ...')
        model.fit(self.df_train[self.lst_col_model], self.df_train[self.col_target])

        print('Predicting ' + str(modeltype) + ' model ...')
        self.df_test['pred'] = model.predict(
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
        df = self.df_test[['Date', 'Eolienne', 'pred']]
        df = df.groupby(['Date', 'Eolienne']).mean().reset_index()
        df.to_csv(fname_out, sep=';', index=False)


    def param_study_grad(self):
        learning_rate_arr = np.arange(0.1, 0.5, 0.1)
        n_estimators_arr = np.arange(400, 500, 100)
        max_depth_arr = np.arange(8, 20, 2)

        for learning_rate in learning_rate_arr:
            for n_estimators in n_estimators_arr:
                for max_depth in max_depth_arr:
                    params = {'learning_rate': learning_rate,
                              'n_estimators': n_estimators,
                              'max_depth': max_depth}
                    self.compute(ensemble.GradientBoostingRegressor, **params)


    def param_study_lst_grid(self,
                             modeltype=ensemble.GradientBoostingRegressor):
        from sklearn.model_selection import train_test_split
        from itertools import combinations
        df_turb = get_df_turbines(lst_turb=[1])
        self.scores = pd.DataFrame(columns=['lst_turb', 'lst_grid',
                                            'Model', 'Params', 'MAE'])
        for n_grid in range(1, 17):
            for lst in combinations(range(1, 17), n_grid):
                self.lst_grid = lst
                df_weather = get_df_weather(lst_grid=lst)
                df = merge_df_turb_weather(df_turb, df_weather)
                self.df_train, self.df_test = train_test_split(df, test_size=0.2)
                self.compute(ensemble.GradientBoostingRegressor, verbose=1)




# Some checks :
if not os.path.isdir(data_dir):
    raise Exception(data_dir + " doesn't exist.")

if not os.path.isdir(data_reformated_dir):
    os.makedirs(data_reformated_dir)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

params_grad = {'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.1,
               'verbose': 1}

# m = model_ml(lst_turb=range(1, 12), lst_grid=range(1,17))
m = model_ml()
m.get_datas()
df = m.df_train[ [m.col_target] + m.lst_col_model ]


# m = model_ml(lst_turb=range(1, 12), lst_grid=[8, 9])
# m = model_ml(submit_mode=True)
# m.compute(LinearRegression)
# m.compute(ensemble.GradientBoostingRegressor, **params_grad)

# df_weather = get_df_weather([1,2])
# df_turb = get_df_turbines([1,3])
