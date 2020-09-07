#!/usr/bin/env python3

import sys
import math
import datetime
import numpy as np
import scipy as sp
import scipy.signal
import scipy.signal.windows
import pandas as pd

from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import labelling


COUNTRIES_TO_PLOT = [
    'United Kingdom',
    'Italy',
    'France',
    'Germany',
    'Spain',
    # 'Diamond Princess',
    # 'China',
    # 'US',
]

DATA_FILES = {
    'confirmed': 'data/time_series_covid19_confirmed_global.csv',
    'deceased': 'data/time_series_covid19_deaths_global.csv',
    'recovered': 'data/time_series_covid19_recovered_global.csv',
}


class PlotParams(object):
    def __init__(self, data=None, label=None, axis_methods=[]):
        self.data = data
        self.label = label
        self.axis_methods = axis_methods


def moving_average(data_set, period=3):
    weights = np.ones(period) / period
    return np.convolve(data_set, weights, mode='valid')


def open_file(filename, column_name):
    dataset = pd.read_csv(filename)
    return (dataset.drop(['Province/State', 'Lat', 'Long'], axis=1)
            .groupby('Country/Region', as_index=False).aggregate('sum')
            .melt(id_vars='Country/Region', var_name='date', value_name=column_name)
            .astype({'date':'datetime64[ns]'}))


def create_dataframe(data_files):
    data_file_generator = ((k, v) for k, v in data_files.items())
    key, filename = next(data_file_generator)
    df = open_file(filename, key)
    for key, filename in data_file_generator:
        df = df.merge(open_file(filename, key))
    return df


def add_case_threshold(df, threshold):
    field = 'days:{}'.format(threshold)
    countries = df['Country/Region'].unique()
    df[field] = np.timedelta64(0)
    for c in countries:
        index = np.argmax(df[df['Country/Region'] == c]['confirmed'] > threshold)
        df.loc[(df['Country/Region'] == c).values, field] = ((
            df[df['Country/Region'] == c]['date'].values - df[df['Country/Region'] == c]['date'].values[index])
            .astype('timedelta64[D]'))


def plot_from_params(plot_params):
    fig, axes = plt.subplots(*plot_params.shape, sharex=True, figsize=(10,20))  # sharex=True
    for params, axis in zip(plot_params.flatten(), axes.flatten()):
        params.data.plot(ax=axis, legend=False)
        axis.set_ylabel(params.label)
        for meth in params.axis_methods:
            meth(axis)
    axes.flatten()[0].legend()  # Only first plot has legend
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.10, right=0.90, top=0.99, bottom=0.03)
    return fig, axes


x_lim = None
df = create_dataframe(DATA_FILES)

case_threshold = 100
relative_dates = False
dfp = df.copy()
dfp = dfp[df['Country/Region'].str.contains('|'.join(COUNTRIES_TO_PLOT))]
if relative_dates:
    add_case_threshold(dfp, case_threshold)
    dfp = dfp.set_index('days:{}'.format(case_threshold))
    dfp = dfp.drop('date', axis=1)
    dfp = dfp.pivot(columns='Country/Region')
    dfp = dfp[dfp.index >= np.timedelta64(0)]
    dfp.index = dfp.index.astype('timedelta64[D]')
if not relative_dates:
    dfp = dfp.set_index('date')
    dfp = dfp.pivot(columns='Country/Region')


plt.style.use(['dark_background'])
sns.set_palette("tab10",plt.cm.tab10.N)

#confirmed_cutoff =  (dfp['confirmed']>float(case_threshold)).any(axis=1).shift(-1, fill_value=True)
plots = np.array([
    [
    PlotParams(
        data=dfp['confirmed'].rolling(7, center=True).mean(),
        label=r'$ N(t) $',
        axis_methods=[
            lambda x: x.set_yscale('log'),
            lambda x: x.grid(which='major', alpha=0.25),
            lambda x: x.grid(which='minor', alpha=0.10),
        ]),
    PlotParams(
        data=dfp['confirmed'].rolling(7, center=True).mean().diff(),
        label=r'$ \frac{dN(t)}{dt} $',
        axis_methods=[
            lambda x: x.set_yscale('log'),
            lambda x: x.grid(which='major', alpha=0.25),
            lambda x: x.grid(which='minor', alpha=0.10),
        ]),
    PlotParams(
        data=dfp['confirmed'].rolling(7, center=True).mean().rolling(21, center=True, win_type='triang').mean().diff().diff(),
        label=r'$ \frac{d^2N(t)}{dt^2} $',
        axis_methods=[
            lambda x: x.set_yscale('symlog', linthreshy=10),
            lambda x: x.grid(which='major', alpha=0.25),
            lambda x: x.grid(which='minor', alpha=0.10),
        ]),
    ], [
    PlotParams(
        data=dfp['deceased'].rolling(7, center=True).mean(), 
        label=r'$ N(t) $',
        axis_methods=[
            lambda x: x.set_yscale('log'),
            lambda x: x.grid(which='major', alpha=0.25),
            lambda x: x.grid(which='minor', alpha=0.10),
            lambda x: x.yaxis.set_label_position("right"),
            lambda x: x.yaxis.tick_right(),
        ]),
    PlotParams(
        data=dfp['deceased'].rolling(7, center=True).mean().diff(),
        label=r'$ \frac{dN(t)}{dt} $',
        axis_methods=[
            lambda x: x.set_yscale('log'),
            lambda x: x.grid(which='major', alpha=0.25),
            lambda x: x.grid(which='minor', alpha=0.10),
            lambda x: x.yaxis.set_label_position("right"),
            lambda x: x.yaxis.tick_right(),
        ]),
    PlotParams(
        data=dfp['deceased'].rolling(7, center=True).mean().rolling(21, center=True, win_type='triang').mean().diff().diff(),
        label=r'$ \frac{d^2N(t)}{dt^2} $',
        axis_methods=[
            lambda x: x.set_yscale('symlog', linthreshy=10),
            lambda x: x.grid(which='major', alpha=0.25),
            lambda x: x.grid(which='minor', alpha=0.10),
            lambda x: x.yaxis.set_label_position("right"),
            lambda x: x.yaxis.tick_right(),
        ]),
    ],
]).transpose()

fig, axs = plot_from_params(plots)
plt.show()
