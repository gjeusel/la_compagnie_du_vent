#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from datetime import datetime, timedelta, tzinfo
from dateutil import tz

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn import ensemble

import plot


##############################################
# Global variables :
script_path = os.path.abspath(sys.argv[0])
working_dir = os.path.dirname(script_path)

data_dir = working_dir + "/data/"
data_reformated_dir = working_dir + "/reformated_data/"
results_dir = working_dir + "/results/"

park_col_type = {
    'drop': ['Etat', 'TurbBrut', 'TurbOK', 'GridBrut', 'GridOK', 'Figee',
             'Manquante'],
    'keep': ['Date', 'Eolienne', 'Production', 'Fonctionnement',
             'Production_mean_hour']
}


# Function from ipython notebook : reading ParcX_20XX.csv
def create_df_park_data(list_num_park, list_date_park):
    df_park_data = pd.DataFrame()
    for num_park in list_num_park:
        for date_park in list_date_park:
            fname_in = data_dir + 'Parc' + str(num_park) + '_'\
                + str(date_park) + '.csv'
            # print('Reading ' + fname_in + '...')
            df_park_data = df_park_data.append(
                pd.read_csv(fname_in, sep=";", decimal=','),
                ignore_index=True)
    return df_park_data


def reformate_park_csv(list_num_park=[1, 2, 3],
                       list_date_park=['2015', '2016'],
                       sep=';'):
    """Read all ParcX_20XX.csv,
    add a Production_mean_hour column, keep only park_col_type['keep'],
    and finally right into csv per turbine in data_reformated_dir.
    """

    # Reading parkX_20XX.csv ...
    df = create_df_park_data(list_num_park, list_date_park)

    # Dropping Useless columns for speed up
    df.drop(park_col_type['drop'], axis=1, inplace=True)

    # Converting in datetime types and keeping in GMT+01:
    print("Converting 'Date' column in datetime type ...")
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M")

    # we create an ident for each hour "Date_hour_int"
    print('Constructing id for each date & hour ...')
    df["Date_hour_int"] = df["Date"].dt.year*10**6 + df["Date"].dt.month*10**4\
        + df["Date"].dt.day*10**2 + df["Date"].dt.hour

    # we create a dataframe with "production_mean_hour" value for each
    # Eolienne*date_hour_int
    print("Computing 'Production_mean_hour' ...")
    df_product_mean = df[df["Fonctionnement"] == 1]\
        .groupby(["Eolienne", "Date_hour_int"])["Production"]\
        .mean().reset_index().rename(columns={"Production": "Production_mean_hour"})

    # we add this value in the initial dataset "df"
    df = pd.merge(df, df_product_mean,
                  on=["Eolienne", "Date_hour_int"], how="left")
    df = df[park_col_type['keep']]

    # output csv files per turbine :
    for num_turb in range(1, 12):
        fname_out = data_reformated_dir + 'turb_' + str(num_turb) + '.csv'
        print('Storing ' + fname_out + ' ...')
        df_tmp = df.loc[df['Eolienne'] == 'Turb'+str(num_turb)]
        df_tmp.to_csv(fname_out, sep=sep, index=False)


def reformate_PrevMeteo_excel(sep=';'):
    for i in range(1, 17):
        fname_in = data_dir + 'PrevMeteo_Grille'+str(i)+'.xlsx'
        fname_out = data_reformated_dir + 'PrevMeteo_Grille'+str(i)+'.csv'
        print('Converting ' + fname_in + ' into ' + fname_out + ' ...')

        df = pd.read_excel(fname_in, sheetname='Feuil1', header=0)
        # Not Needed to change datetimes because already in GMT+00

        # The aim of this challenge is to predict the Production of tomorrow.
        # So only forecast (fc_hor) between 25h and 47h will be kept :
        df = df[(df['fc_hor'] >= 24) & (df['fc_hor'] <= 47)]

        # No Daylight Saving Time (DST) gap so assumed in GMT+00
        # Convert datetime GMT+00 into datetime GMT+01:
        # Get a SegFault on Linux :
        # df['date'] = df['date'].apply(lambda x: datetime
        #                             .strptime(x, '%d/%m/%Y %H:%M')
        #                             .replace(tzinfo=tz.gettz('GMT+00'))
        #                             .astimezone(tz.gettz('GMT+01')))
        df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
        df['date'] = df['date'].apply(lambda x: x + timedelta(hours=1))

        # Keeping only registers for which datetime <= 14/04/2017 23:00 (of no use)
        df = df[df['date'] <= datetime(2017, 4, 14, 23)]

        # Renaming for same column label as ParcX_20XX.csv :
        df = df.rename(columns={'date': 'Date'})

        df.to_csv(fname_out, sep=sep, index=False)


def reformate_all_data():
    reformate_park_csv(list_num_park=[1, 2, 3],
                       list_date_park=['2015', '2016'])

    reformate_PrevMeteo_excel()
    return


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
    df_turb.sort_values('Date', ascending=1, inplace=True)
    df_turb.reset_index()
    return df_turb


def get_df_weather(lst_grid):
    df_weather = pd.DataFrame()
    for num_grid in lst_grid:
        fname_in_grid = data_reformated_dir + 'PrevMeteo_Grille' +\
                        str(num_grid) + '.csv'
        df_tmp = pd.read_csv(fname_in_grid, sep=';')
        df_tmp['grid id'] = pd.Series(num_grid, index=df_tmp.index)

        df_weather = df_weather.append(df_tmp)
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], format="%Y-%m-%d %H:%M:%S")
    df_weather.sort_values('Date', ascending=1, inplace=True)
    df_weather.reset_index()
    return df_weather


def get_bounds_datetime(df_turb, df_weather):
    # Youngest admissible to avoid missing values :
    dt_min = max(min(df_turb['Date']), min(df_weather['Date']))
    # Oldest admissible to avoid missing values :
    dt_max = min(max(df_turb['Date']), max(df_weather['Date']))
    return dt_min, dt_max


def merge_df_turb_weather(df_turb, df_weather):
    dt_min, dt_max = get_bounds_datetime(df_turb, df_weather)
    df_turb = df_turb[(df_turb['Date'] >= dt_min) & (df_turb['Date'] <=
                      dt_max)]
    df_weather = df_weather[(df_weather['Date'] >= dt_min) &
                            (df_weather['Date'] <= dt_max)]
    df = pd.merge(df_turb, df_weather, on='Date', how='left')
    df.sort_values('Date', ascending=1, inplace=True)
    df.reset_index()
    return df


def get_df_all(lst_turb, lst_grid):
    df_turb = get_df_turbines(lst_turb=lst_turb)
    df_weather = get_df_weather(lst_grid=lst_grid)
    df = merge_df_turb_weather(df_turb, df_weather)
    return df


lst_col_model = ['fc_hor', 'RS', 'CAPE', 'SP', 'CP',
                 'BLD', 'SSHF', 'SLHF', 'MSL', 'BLH', 'TCC', 'U10', 'V10', 'T2',
                 'D2', 'SSRD', 'STRD', 'SSR', 'STR', 'TSR', 'LCC', 'MCC', 'HCC',
                 'TSRC', 'SSRC', 'STRC', 'TP', 'FA', 'U100', 'V100', 'vit_100',
                 'vit_10', 'dir_100', 'dir_10']


lst_col_prev = ['Date', 'Eolienne', 'pred']

dt_start_pred = datetime(2017, 1, 1, 0, 0)
dt_start_pred = datetime(2017, 4, 14, 23, 0)


class model_ml:
    def __init__(self, lst_turb=[1], lst_grid=[8, 9], lst_col_model=lst_col_model,
                 col_target='Production_mean_hour',
                 submit_mode=False):

        # Storing config :
        self.submit_mode = submit_mode

        if submit_mode:
            print('---> Submit mode activated ! <---')
            self.lst_turb = range(1, 12)
        else:
            self.lst_turb = lst_turb

        self.lst_grid = lst_grid
        self.lst_col_model = lst_col_model
        self.col_target = col_target

    def get_datas(self):
        print('Reading and formatting all datas ...')
        if self.submit_mode:
            df_parks_2017 = create_df_park_data(list_num_park=[1, 2, 3],
                                                list_date_park=['2017'])
            df_parks_2017['Date'] = pd.to_datetime(
                df_parks_2017['Date'], format="%d/%m/%Y %H:%M")

            df_parks_2017 = df_parks_2017[
                (df_parks_2017['Date'].dt.minute == 0) &
                (df_parks_2017['Fonctionnement'] == 1)]
            df_parks_2017 = df_parks_2017[['Date', 'Eolienne']]

            df_turb = get_df_turbines(lst_turb=range(1, 12))
            df_weather = get_df_weather(lst_grid=self.lst_grid)

            self.df_train = merge_df_turb_weather(df_turb, df_weather)
            self.df_pred = merge_df_turb_weather(df_parks_2017, df_weather)

        else:
            df = get_df_all(lst_turb=self.lst_turb, lst_grid=self.lst_grid)
            from sklearn.model_selection import train_test_split
            self.df_train, self.df_test = train_test_split(df, test_size=0.2)
            self.scores = pd.DataFrame(columns=['lst_turb', 'lst_grid',
                                                'Model', 'Params', 'MAE'])

    def compute(self, modeltype, **kwargs):
        print('-------------------------------------------')
        if kwargs != {}:
            print(kwargs)

        if self.submit_mode:
            self.compute_submit(modeltype=modeltype, **kwargs)
        else:
            model = modeltype(**kwargs)

            print('Training ' + str(modeltype) + ' model ...')
            model.fit(self.df_train[self.lst_col_model], self.df_train[self.col_target])

            print('Testing ' + str(modeltype) + ' model ...')
            y_pred = model.predict(self.df_test[self.lst_col_model])
            mae = mean_absolute_error(y_pred, self.df_test[self.col_target])
            self.scores.loc[self.scores.shape[0]] =\
                [self.lst_turb, self.lst_grid, str(modeltype), kwargs, mae]
            return mae

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



    def compute_submit(self, modeltype, **kwargs):
        model = modeltype(**kwargs)
        print('Training ' + str(modeltype) + ' model ...')
        model.fit(self.df_train[self.lst_col_model], self.df_train[self.col_target])

        print('Predicting ' + str(modeltype) + ' model ...')
        self.df_pred['pred'] = model.predict(self.df_pred[self.lst_col_model])

    def write_submit_csv(self):
        now = datetime.now()
        fname_out = results_dir + 'lcv_submit_GJ_' + str(now.day) +\
            '_' + str(now.hour) + '_' + str(now.minute) + '.csv'
        # Computing the mean production predicted for each hour according to
        # grid id :
        df = self.df_pred[['Date', 'Eolienne', 'pred']]
        df = df.groupby(['Date', 'Eolienne']).mean().reset_index()
        df.to_csv(fname_out, sep=';', index=False)


class test_ipython:
    def __init__(self):
        if not os.path.isdir(data_reformated_dir):
            return 'Missing ' + data_reformated_dir

        self.turb1 = pd.read_csv(data_reformated_dir+'turb_1.csv', sep=';')
        self.turb1['Date'] = pd.to_datetime(self.turb1['Date'],
                                            format="%Y-%m-%d %H:%M:%S")

        self.grid1 = pd.read_csv(data_reformated_dir+'PrevMeteo_Grille1.csv',
                                 sep=';')
        self.grid1['Date'] = pd.to_datetime(self.grid1['Date'],
                                            format="%Y-%m-%d %H:%M:%S")


# Some checks :
if not os.path.isdir(data_dir):
    raise Exception(data_dir + " doesn't exist.")

if not os.path.isdir(data_reformated_dir):
    os.makedirs(data_reformated_dir)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

params_grad = {'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.1,
               'verbose': 1}

m = model_ml()
# m = model_ml(lst_turb=range(1, 12), lst_grid=[8, 9])
# m = model_ml(submit_mode=True)
# m.compute(LinearRegression)
# m.compute(ensemble.GradientBoostingRegressor, **params_grad)
