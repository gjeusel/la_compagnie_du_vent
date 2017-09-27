#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from datetime import datetime, timedelta, tzinfo
from dateutil import tz

import pandas as pd

import numpy as np

from sklearn.metrics import mean_absolute_error


##############################################
# Global variables :
script_path = os.path.abspath(sys.argv[0])
working_dir = os.path.dirname(script_path)

data_dir = working_dir + "/data/"
data_reformated_dir = working_dir + "/reformated_data/"

park_col_type = {
    'float': ['Vent', 'P', 'Production', 'Vitesse génératrice', 'Température\
              génératrice'],
    'int': ['State', 'Turb', 'Fonctionnement'],  # State is the same info as Etat
    'str': ['Eolienne', 'Etat', 'Catégorie'],

    'drop': ['Etat', 'TurbBrut', 'TurbOK', 'GridBrut', 'GridOK', 'Figee',
             'Manquante'],
    'keep': ['date', 'Eolienne', 'Production', 'Fonctionnement',
             'Production_mean_hour']
}


def reformate_park_csv(list_num_park=[1, 2, 3],
                       list_date_park=['2015', '2016'],
                       sep=';'):
    """Read all ParcX_20XX.csv, convert datetime GMT+01 into datetime GMT+00,
    add a Production_mean_hour column, keep only park_col_type['keep'],
    and finally right into csv per turbine in data_reformated_dir.
    """

    # Function from ipython notebook :
    # Reading all park csv
    def create_df_park_data(list_num_park, list_date_park):
        df_park_data = pd.DataFrame()
        for num_park in list_num_park:
            for date_park in list_date_park:
                fname_in = data_dir + 'Parc' + str(num_park) + '_'\
                    + str(date_park) + '.csv'
                print('Reading ' + fname_in + '...')
                df_park_data = df_park_data.append(
                    pd.read_csv(fname_in, sep=";", decimal=','),
                    ignore_index=True)
        return df_park_data

    # Reading parkX_20XX.csv ...
    df = create_df_park_data(list_num_park, list_date_park)

    # Dropping Useless columns for speed up
    df.drop(park_col_type['drop'], axis=1, inplace=True)

    # Renaming for same column label as Prevweather_Grille :
    df = df.rename(columns={'Date': 'date'})

    # Changing data types :
    # No Daylight Saving Time (DST) gap so assumed in GMT+01
    # and switched to GMT+00
    print('Reformatting dates ...')
    # Get a SegFault on Linux :
    # df['date'] = df['date'].apply(lambda x: datetime
    #                             .strptime(x, '%d/%m/%Y %H:%M')
    #                             .replace(tzinfo=tz.gettz('GMT+01'))
    #                             .astimezone(tz.gettz('GMT+00')))
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M")
    df['date'] = df['date'].apply(lambda x: x - timedelta(hours=1))

    # we create an ident for each hour "Date_hour_int"
    print('Constructing id for each date & hour ...')
    df["Date_hour_int"] = df["date"].dt.year*10**6 + df["date"].dt.month*10**4\
        + df["date"].dt.day*10**2 + df["date"].dt.hour

    # we create a dataframe with "production_mean_hour" value for each
    # Eolienne*date_hour_int
    print("Computing 'Production_mean_hour' ...")
    df_product_mean = df[df["Fonctionnement"] == 1]\
        .groupby(["Eolienne", "Date_hour_int"])["Production"]\
        .mean().reset_index().rename(columns={"Production": "Production_mean_hour"})

    # we add this value in the initial dataset "df"
    df = pd.merge(df, df_product_mean,
                            on=["Eolienne", "Date_hour_int"], how="left")
    df = df[park_col_type['keep']].copy()

    # output csv files per turbine :
    for num_turb in range(1, 12):
        fname_out = data_reformated_dir + 'turb_' + str(num_turb) + '.csv'
        print('Storing ' + fname_out + ' ...')
        df_tmp = df.loc[df['Eolienne'] == 'Turb'+str(num_turb)]
        df_tmp.to_csv(fname_out, sep=sep, index=False)


def reformate_Prevweather_excel(sep=';'):
    for i in np.arange(1, 17):
        fname_in = 'Prevweather_Grille'+str(i)+'.xlsx'
        fname_out = 'Prevweather_Grille'+str(i)+'.csv'
        print('Converting ' + fname_in + ' into ' + fname_out + ' ...')

        df = pd.read_excel(data_dir+fname_in, sheetname='Feuil1', header=0)
        # Not Needed to change datetimes because already in GMT+00

        # The aim of this challenge is to predict the Production of tomorrow.
        # So only forecast (fc_hor) between 25h and 47h will be kept :
        df = df[(df['fc_hor']>=24) & (df['fc_hor']<=47)].copy()

        df.to_csv(data_reformated_dir+fname_out, sep=sep, index=False)


def reformate_all_data():
    reformate_park_csv(list_num_park=[1, 2, 3],
                       list_date_park=['2015', '2016'])

    reformate_Prevweather_excel()
    pass


def get_df_all_per_turbine(num_turb=1, lst_grid=range(1, 17)):
    fname_in_turb = data_reformated_dir + 'turb_' + str(num_turb) + '.csv'
    df_turb = pd.read_csv(fname_in_turb, sep=';')
    df_turb['date'] = pd.to_datetime(df_turb['date'], format="%Y-%m-%d %H:%M:%S")

    # !TO DO : better the model by splitting according to Fonctionnement per
    # minute
    # Right now, only keeping dt.minute == 0 :

    df_turb = df_turb[(df_turb['date'].dt.minute == 0)
                      & (df_turb['Fonctionnement'] == 1)].copy()

    df_weather = pd.DataFrame()
    for num_grid in lst_grid:
        fname_in_grid = data_reformated_dir + 'PrevMeteo_Grille' +\
                        str(num_grid) + '.csv'
        df_weather = df_weather.append(pd.read_csv(fname_in_grid, sep=';'))
    df_weather['date'] = pd.to_datetime(df_weather['date'],
                                        format="%Y-%m-%d %H:%M:%S")

    def get_bounds_datetime(df_turb, df_weather):
        # Youngest admissible to avoid missing values :
        dt_min = max(min(df_turb['date']), min(df_weather['date']))
        # Oldest admissible to avoid missing values :
        dt_max = min(max(df_turb['date']), max(df_weather['date']))
        return dt_min, dt_max

    dt_min, dt_max = get_bounds_datetime(df_turb, df_weather)
    df_turb = df_turb[(df_turb['date'] > dt_min) & (df_turb['date'] <
                      dt_max)].copy()
    df_weather = df_weather[(df_weather['date'] > dt_min) & (df_weather['date'] <
                            dt_max)].copy()

    df = pd.merge(df_turb, df_weather, on='date', how='left')
    df.sort_values('date', ascending=1, inplace=True)
    return df


lst_col_model = ['fc_hor', 'RS', 'CAPE', 'SP', 'CP',
                 'BLD', 'SSHF', 'SLHF', 'MSL', 'BLH', 'TCC', 'U10', 'V10', 'T2',
                 'D2', 'SSRD', 'STRD', 'SSR', 'STR', 'TSR', 'LCC', 'MCC', 'HCC',
                 'TSRC', 'SSRC', 'STRC', 'TP', 'FA', 'U100', 'V100', 'vit_100',
                 'vit_10', 'dir_100', 'dir_10']


class model_ml:
    def __init__(self, num_turb=1, lst_grid=[1], lst_col_model=lst_col_model,
                 col_target='Production_mean_hour'):
        from sklearn.model_selection import train_test_split
        df = get_df_all_per_turbine(num_turb=num_turb, lst_grid=lst_grid)
        self.train, self.test = train_test_split(df, test_size=0.2)
        self.lst_col_model = lst_col_model
        self.col_target = col_target
        self.scores = pd.DataFrame(columns=['Model', 'Params', 'MAE'])

    def linear_reg(self):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        print('Training LinearRegression model ...')
        model.fit(self.train[self.lst_col_model], self.train[self.col_target])

        print('Testing LinearRegression model ...')
        y_pred = model.predict(self.test[self.lst_col_model])
        mae = mean_absolute_error(y_pred, self.test[self.col_target])
        self.scores.loc[self.scores.shape[0]] = ['LinearRegression', '', mae]
        return mae


class test_ipython:
    def __init__(self):
        if not os.path.isdir(data_reformated_dir):
            return 'Missing ' + data_reformated_dir

        self.turb1 = pd.read_csv(data_reformated_dir+'turb_1.csv', sep=';')
        self.turb1['date'] = pd.to_datetime(self.turb1['date'],
                                            format="%Y-%m-%d %H:%M:%S")

        self.grid1 = pd.read_csv(data_reformated_dir+'PrevMeteo_Grille1.csv',
                                   sep=';')
        self.grid1['date'] = pd.to_datetime(self.grid1['date'],
                                            format="%Y-%m-%d %H:%M:%S")


# Some checks :
if not os.path.isdir(data_dir):
    raise Exception(data_dir + " doesn't exist.")

if not os.path.isdir(data_reformated_dir):
    os.makedirs(data_reformated_dir)

m = model_ml()
