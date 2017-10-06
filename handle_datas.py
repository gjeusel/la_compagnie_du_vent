#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from datetime import datetime, timedelta, tzinfo
from dateutil import tz

import pandas as pd
import numpy as np

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

dt_start_pred = datetime(2017, 1, 1, 0, 0)
dt_end_pred = datetime(2017, 4, 14, 23, 0)

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
        fname_out_2017 = data_reformated_dir + 'PrevMeteo_Grille'+str(i) +\
            '_2017'+'.csv'
        print('Converting ' + fname_in + ' into :\n' + fname_out +
              '\n' + fname_out_2017 + '...')

        df = pd.read_excel(fname_in, sheetname='Feuil1', header=0)
        # Not Needed to change datetimes because already in GMT+00

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
        df = df[df['date'] <= dt_end_pred]

        # Renaming for same column label as ParcX_20XX.csv :
        df = df.rename(columns={'date': 'Date'})

        # 2015 - 2016 :
        df_2015_2016 = df[df['Date'] < dt_start_pred]
        df_2015_2016.to_csv(fname_out, sep=sep, index=False)

        # 2017
        df_2017 = df[df['Date'] >= dt_start_pred]
        df_2017.to_csv(fname_out_2017, sep=sep, index=False)


def reformate_all_data():
    reformate_park_csv(list_num_park=[1, 2, 3],
                       list_date_park=['2015', '2016'])

    reformate_PrevMeteo_excel()
    return



