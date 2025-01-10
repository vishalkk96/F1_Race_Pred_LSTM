import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.api import VAR
pd.set_option('display.max_columns', None)

def read_data(data_df):
    df = data_df.copy()

    timedelta_columns = ['Session_Time', 'Sector_1', 'Sector_2', 'Sector_3', 'Race_Laptime', 'Race_Pit_In', 'Race_Pit_Out']
    for col in timedelta_columns:
        df[col] = pd.to_timedelta(df[col])

    date_columns = ['Date', 'Race Date', 'DOB']  # List of columns to parse

    for col in date_columns:
        # Create a new column for parsed dates
        new_col_name = col + '_Col'
        df[new_col_name] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')

        # Attempt to parse with the second format where the first resulted in NaT
        df.loc[df[new_col_name].isna(), new_col_name] = pd.to_datetime(df.loc[df[new_col_name].isna(), col], format='%Y-%m-%d', errors='coerce')

        # Replace the original column with the new parsed column
        df[col] = df[new_col_name]

        # Drop the temporary parsed column
        df.drop(new_col_name, axis=1, inplace=True)

        # Convert columns to timedelta
    timedelta_columns = ['Session_Time', 'Sector_1', 'Sector_2', 'Sector_3', 'Race_Laptime', 'Race_Pit_In', 'Race_Pit_Out']
    for col in timedelta_columns:
        df[col] = pd.to_timedelta(df[col])

    datetime_columns = ['Race_Lap_Start', 'Race_Lap_End']

    for col in datetime_columns:
        # Create a new column for parsed dates
        new_col_name = col + '_Col'
        df[new_col_name] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

        # Attempt to parse with the second format where the first resulted in NaT
        df.loc[df[new_col_name].isna(), new_col_name] = pd.to_datetime(df.loc[df[new_col_name].isna(), col], format='%d-%m-%Y %H:%M:%S.%f', errors='coerce')

        # Replace the original column with the new parsed column
        df[col] = df[new_col_name]

        # Drop the temporary parsed column
        df.drop(new_col_name, axis=1, inplace=True)

    df = df.loc[df['Lap_Num'] > 1].reset_index(drop = True)
    df = df.loc[df['DNF Lap'] == 0].reset_index(drop = True)
    df = df.loc[~df['Qual_Sec1_Spd'].isna()].reset_index(drop = True)
    df = df.loc[~df['Race_Sec1_Spd'].isna()].reset_index(drop = True)

    df = df.groupby(['Date', 'Driver']).filter(lambda x: len(x) >= 40)

    return df

def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value, usedlag, nobs, critical_values, icbest = result
    
    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values
    }


def adf_data_eval(df):
    stat_cols = ["Sec1_Diff", "Sec2_Diff", "Sec3_Diff"]

    # Store results in a list or dictionary
    results_list = []

    # Group by each combination of Date and Driver
    for (group_keys, group_df) in df.groupby(["Date", "Race", "Driver"]):
        date_val, race_val, driver_val = group_keys
        
        for col in stat_cols:
            # Perform ADF test
            adf_results = adf_test(group_df[col])
            
            # Save the results
            results_list.append({
                "Date": date_val,
                "Race": race_val,
                "Driver": driver_val,
                "Column": col,
                "ADF_Statistic": adf_results["adf_stat"],
                "p_value": adf_results["p_value"]
            })

    # Convert results to a DataFrame for a cleaner look
    results_df = pd.DataFrame(results_list)

    return results_df

def pacf_calc(df):
    # Columns for which we'll compute PACF
    stat_cols = ["Sec1_Diff", "Sec2_Diff", "Sec3_Diff"]

    # A list to collect all rows of our final pacf data
    pacf_results = []

    # Group by (Date, Driver)
    for (date_val, race_val, driver_val), group_df in df.groupby(["Date", "Race", "Driver"]):
        
        # Drop NAs within each group (optional, but recommended)
        group_df_clean = group_df[stat_cols].dropna()
        
        if len(group_df_clean) < 2:
            continue
        
        # Compute PACF for each column, up to lag=20.
        sec1_pacf = pacf(group_df_clean["Sec1_Diff"], nlags=5, method='ywm')
        sec2_pacf = pacf(group_df_clean["Sec2_Diff"], nlags=5, method='ywm')
        sec3_pacf = pacf(group_df_clean["Sec3_Diff"], nlags=5, method='ywm')
                
        for lag in range(len(sec1_pacf)):  # This goes from 0 to 20
            row = {
                "Date": date_val,
                "Race": race_val,
                "Driver": driver_val,
                "Lag": lag,
                "Sec1_Pacf": sec1_pacf[lag],
                "Sec2_Pacf": sec2_pacf[lag],
                "Sec3_Pacf": sec3_pacf[lag],
            }
            pacf_results.append(row)

    # Convert pacf_results list into a DataFrame
    pacf_df = pd.DataFrame(pacf_results)

    lag_df = pacf_df.loc[(pacf_df['Lag'] > 0)].groupby('Lag').agg({'Sec1_Pacf' : 'max', 'Sec2_Pacf' : 'max', 'Sec3_Pacf' : 'max'}).reset_index()

    lag_df[['Sec1_Pacf', 'Sec2_Pacf', 'Sec3_Pacf']] = lag_df[['Sec1_Pacf', 'Sec2_Pacf', 'Sec3_Pacf']].to_numpy().round(2)

    return lag_df

def accumulate_norm_pred(group):
        group = group.sort_values('Lap_Num').copy()

        # We'll iterate from the 2nd row onward, adding the diff to the previous Norm_Pred
        for i in range(19, len(group)):
            prev_idx = group.index[i - 1]
            curr_idx = group.index[i]

            group.loc[curr_idx, 'Sec1_Norm_Pred'] = (
                group.loc[prev_idx, 'Sec1_Norm_Pred'] 
                + group.loc[curr_idx, 'Sec1_Diff_Pred']
            )

            group.loc[curr_idx, 'Sec2_Norm_Pred'] = (
                group.loc[prev_idx, 'Sec2_Norm_Pred']
                + group.loc[curr_idx, 'Sec2_Diff_Pred']
            )

            group.loc[curr_idx, 'Sec3_Norm_Pred'] = (
                group.loc[prev_idx, 'Sec3_Norm_Pred']
                + group.loc[curr_idx, 'Sec3_Diff_Pred']
            )

        return group


def var(df: pd.DataFrame, nlaps: int):
    df = df.copy()  
    df.sort_values(by=['Date', 'Race', 'Driver', 'Lap_Num'], inplace=True)

    var_cols = ['Sec1_Diff', 'Sec2_Diff', 'Sec3_Diff']
    
    # Create placeholders for predicted columns
    df['Sec1_Diff_Pred'] = np.nan
    df['Sec2_Diff_Pred'] = np.nan
    df['Sec3_Diff_Pred'] = np.nan

    # Group by the time-series definition: (Date, Race, Driver)
    grouped = df.groupby(['Date', 'Race', 'Driver'], group_keys=False)

    def fit_and_forecast(group):
        # Separate training (Lap_Num <= 20) and forecast (Lap_Num > 20)
        train_data = group[group['Lap_Num'] <= 20].copy()
        forecast_data = group[group['Lap_Num'] > 20].copy()

        # If there's not enough data to train, just copy actual values as "predicted"
        if len(train_data) <= nlaps:
            group['Sec1_Diff_Pred'] = group['Sec1_Diff']
            group['Sec2_Diff_Pred'] = group['Sec2_Diff']
            group['Sec3_Diff_Pred'] = group['Sec3_Diff']
            return group

        # Prepare training data (drop rows with missing data in var_cols)
        train_data_clean = train_data[var_cols].dropna()
        if train_data_clean.empty:
            # If training data is empty after dropping NAs, fallback to actual
            group['Sec1_Diff_Pred'] = group['Sec1_Diff']
            group['Sec2_Diff_Pred'] = group['Sec2_Diff']
            group['Sec3_Diff_Pred'] = group['Sec3_Diff']
            return group

        # 2. Fit the VAR model on var_cols
        model = VAR(train_data_clean)
        fitted_model = model.fit(maxlags=nlaps)  # changed from 'nlags' to 'maxlags'

        # 3. In-sample: for Lap_Num <= 20, copy actual to predicted
        group.loc[group['Lap_Num'] <= 20, 'Sec1_Diff_Pred'] = group['Sec1_Diff']
        group.loc[group['Lap_Num'] <= 20, 'Sec2_Diff_Pred'] = group['Sec2_Diff']
        group.loc[group['Lap_Num'] <= 20, 'Sec3_Diff_Pred'] = group['Sec3_Diff']

        # 4. Out-of-sample forecast for Lap_Num > 20
        if not forecast_data.empty:
            n_forecasts = len(forecast_data)
            forecast_input = train_data_clean.values[-fitted_model.k_ar:]
            predicted = fitted_model.forecast(y=forecast_input, steps=n_forecasts)

            # Map var_cols to their index in the predicted array
            forecast_cols_idx = {col: i for i, col in enumerate(var_cols)}

            # Assign forecasts to forecast_data
            forecast_data_idx = forecast_data.index
            for step, idx in enumerate(forecast_data_idx):
                forecast_data.loc[idx, 'Sec1_Diff_Pred'] = predicted[step, forecast_cols_idx['Sec1_Diff']]
                forecast_data.loc[idx, 'Sec2_Diff_Pred'] = predicted[step, forecast_cols_idx['Sec2_Diff']]
                forecast_data.loc[idx, 'Sec3_Diff_Pred'] = predicted[step, forecast_cols_idx['Sec3_Diff']]

            # Update the group with forecasted values
            group.update(forecast_data)

        return group

    # Apply the function to each group
    df_modified = grouped.apply(fit_and_forecast)

    df_modified['Sec1_Norm_Pred'] = np.where(
        df_modified['Lap_Num'] <= 20, df_modified['Sec1_Norm'], np.nan
    )
    df_modified['Sec2_Norm_Pred'] = np.where(
        df_modified['Lap_Num'] <= 20, df_modified['Sec2_Norm'], np.nan
    )
    df_modified['Sec3_Norm_Pred'] = np.where(
        df_modified['Lap_Num'] <= 20, df_modified['Sec3_Norm'], np.nan
    )

    # Define a helper function to accumulate each group's Norm_Pred
    

    # Accumulate Norm_Pred within each (Date, Race, Driver) group
    df_modified = df_modified.groupby(['Date', 'Race', 'Driver'], group_keys=False).apply(accumulate_norm_pred)

   
    df_modified['Race_Sec1_Spd_Pred'] = df_modified['Sec1_Norm_Pred'] * df_modified['Qual_Sec1_Spd']
    df_modified['Race_Sec2_Spd_Pred'] = df_modified['Sec2_Norm_Pred'] * df_modified['Qual_Sec2_Spd']
    df_modified['Race_Sec3_Spd_Pred'] = df_modified['Sec3_Norm_Pred'] * df_modified['Qual_Sec3_Spd']

   
    df_modified.drop(columns=[
        'Sec1_Norm_Pred', 'Sec2_Norm_Pred', 'Sec3_Norm_Pred',  # newly created intermediate
        'Sec1_Norm', 'Sec2_Norm', 'Sec3_Norm',                # original norm
        'Sec1_Diff_Pred', 'Sec2_Diff_Pred', 'Sec3_Diff_Pred', # predicted diffs
        'Sec1_Diff', 'Sec2_Diff', 'Sec3_Diff'                 # original diffs
    ], inplace=True, errors='ignore')
    
    df_modified['Real_Sec_Spd'] = (df_modified['Race_Sec1_Spd'] + df_modified['Race_Sec2_Spd'] + df_modified['Race_Sec3_Spd'])/3
    df_modified['Pred_Sec_Spd'] = (df_modified['Race_Sec1_Spd_Pred'] + df_modified['Race_Sec2_Spd_Pred'] + df_modified['Race_Sec3_Spd_Pred'])/3

    df_modified['Err_Drv_Spd'] = (100 * (df_modified['Pred_Sec_Spd'] - df_modified['Real_Sec_Spd'])/(df_modified['Real_Sec_Spd']))

    return df_modified
