#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_URL="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

mkdir -p ${DIR}/data
for f in time_series_covid19_confirmed_global.csv time_series_covid19_deaths_global.csv time_series_covid19_recovered_global.csv; do
    wget ${BASE_URL}$f -O ${DIR}/data/$f
done
