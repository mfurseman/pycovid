#!/usr/bin/env python3

import sys
import math
import numpy as np
import pandas as pd

from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import labelling


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


def plot_helper(ax, x_lim, dataset_1, dataset_2, alpha=1.0, label=""):
    dataset_1.plot(ax=ax, linewidth=3,)
    ax.set_prop_cycle(None)  # Reset colours
    dataset_2.plot(ax=ax, linewidth=3, linestyle='--', legend=False, alpha=alpha)
    ax.set_ylabel(label)
    ax.grid(True, which='both', alpha=0.1)
    ax.set_xlim(0, x_lim)


x_lim = None
threshold = 100
df = create_dataframe(DATA_FILES)
add_case_threshold(df, threshold)

dfp = df
dfp = dfp[df['Country/Region'].str.contains('|'.join(COUNTRIES_TO_PLOT))]
dfp = dfp.set_index('days:100')
dfp = dfp.drop('date', axis=1)
dfp = dfp.pivot(columns='Country/Region')
dfp = dfp[dfp.index >= np.timedelta64(0)]
dfp.index = dfp.index.astype('timedelta64[D]')


sns.set()
# sns.set_style('darkgrid')
plt.style.use(['dark_background'])
sns.set_palette("tab10",plt.cm.tab10.N )

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(10,20))
plt.subplots_adjust(wspace=0, hspace=0, left=0.08, right=0.99, top=0.99, bottom=0.03)

plot_helper(ax1, x_lim, 
    dfp['confirmed'],
    dfp['deceased'],
    label=r'Confirmed / Deceased Cumulative Cases     $\left [ C(t) = R_o^t \right ]$')
ax1.set_yscale('log')
ax1.set_ylim(100, 1E5)
ax1.xaxis.set_major_locator(plt.MultipleLocator(1))

plot_helper(ax2, x_lim,
    dfp['confirmed'].apply(np.gradient),
    dfp['deceased'].apply(np.gradient),
    label=r'Confirmed / Deceased Daily Cases      $\left [ \frac{dC(t)}{dt} = t R_0^t \right ]$')
ax2.set_yscale('log')

plot_helper(ax3, x_lim,
    100 * dfp['confirmed'].pct_change(limit=1).apply(moving_average, period=5),
    100 * dfp['confirmed'].pct_change(limit=1),
    alpha=0.5, label="Day on Day %")

plot_helper(ax4, x_lim,
    dfp['confirmed'].apply(lambda x: np.exp(moving_average(np.gradient(np.log(x)), period=5))),
    dfp['confirmed'].apply(lambda x: np.exp(np.gradient(np.log(x)))),
    alpha=0.5, 
    label=r'Growth rate, five point moving averge (confirmed)     $\left [ R_0 = \frac{d ln(C(t))}{dt} \right ]$')

plot_helper(ax5, x_lim,
    dfp['deceased'].apply(lambda x: np.exp(moving_average(np.gradient(np.log(x)), period=5))),
    dfp['deceased'].apply(lambda x: np.exp(np.gradient(np.log(x)))),
    alpha=0.5, 
    label=r'Growth rate, five point moving averge (deceased)     $\left [ R_0 = \frac{d ln(C(t))}{dt} \right ]$')

plot_helper(ax6, x_lim,
    dfp['confirmed'].apply(lambda x: np.gradient(np.exp(moving_average(np.gradient(np.log(x)), period=5)))),
    dfp['confirmed'].apply(lambda x: np.gradient(np.exp(np.gradient(np.log(x))))),
    alpha=0.5, 
    label=r'Gradient of growth rate (confirmed)     $\left [ \frac{dR_0}{dt} = \frac{d^2 ln(C(t))}{dt^2} \right ]$')
ax6.set_yscale('symlog', basey=10, linthreshy=0.01, subsy=range(100))
ax6.set_xlabel("Days since 100 cases")


plt.show()
