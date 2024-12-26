import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

def preprocess_spd_model(df):
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

    df['Driver Points Scored'] = df['Driver Points After'] - df['Driver Points Before']
    df['Track_Status'] = df['Track_Status'].clip(upper=8)
    df = df.loc[(df['DNF Lap'] == 0) & (df['Lap_Num'] > 1) &
                (~df['Sector_1'].isna())].sort_values(by = ['Date', 'Driver Points Scored', 'Driver', 'Lap_Num'],
                                                        ascending = [True, False, True, True]).reset_index(drop = True)

    drvr = df[['Race', 'Date', 'Driver', 'Lap_Num', 'Position', 'Inlap', 'Outlap', 'Tot_Laps', 'Race_Tyre_Type', 'Race_Tyre_Life', 'Race_Sec1_Spd', 'Race_Sec2_Spd', 'Race_Sec3_Spd', 'Driver Points Before', 'Team Points Before', 'Grid Points Before']]
    drvr.columns = ['Race', 'Date', 'Driver', 'Lap', 'Position', 'Inlap', 'Outlap', 'Total', 'Drv_Tyre', 'Drv_Tyre_Life', 'Drv_Sec1_Spd', 'Drv_Sec2_Spd', 'Drv_Sec3_Spd', 'Driver Points Before', 'Team Points Before', 'Grid Points Before']

    race = df.groupby(['Race', 'Date', 'Lap_Num', 'Tot_Laps']).agg({'Race_Sec1_Spd' : 'mean', 'Race_Sec2_Spd' : 'mean', 'Race_Sec3_Spd' : 'mean',
            'Track_Status' : 'max', 'Race_Tyre_Life' : 'mean', 'Driver' : 'nunique'}).reset_index().sort_values(by = ['Date', 'Lap_Num']).reset_index(drop = True)

    tire = df.groupby(['Race', 'Date', 'Lap_Num', 'Tot_Laps' ,'Race_Tyre_Type']).agg({'Driver' : 'nunique'}).reset_index().sort_values(by = ['Date', 'Lap_Num']).reset_index(drop = True)

    tire.columns = ['Race', 'Date', 'Lap_Num', 'Tot_Laps' ,'Tire', 'Drivers']

    hard = tire.loc[tire['Tire'] == 'HARD'].reset_index(drop = True)
    soft = tire.loc[tire['Tire'] == 'SOFT'].reset_index(drop = True)
    medm = tire.loc[tire['Tire'] == 'MEDIUM'].reset_index(drop = True)
    wets = tire.loc[tire['Tire'] == 'WET'].reset_index(drop = True)
    intr = tire.loc[tire['Tire'] == 'INTERMEDIATE'].reset_index(drop = True)

    hard = hard.drop(columns=['Tire'])
    soft = soft.drop(columns=['Tire'])
    medm = medm.drop(columns=['Tire'])
    wets = wets.drop(columns=['Tire'])
    intr = intr.drop(columns=['Tire'])

    hard.columns = ['Race', 'Date', 'Lap_Num', 'Tot_Laps', 'Hards']
    soft.columns = ['Race', 'Date', 'Lap_Num', 'Tot_Laps', 'Softs']
    medm.columns = ['Race', 'Date', 'Lap_Num', 'Tot_Laps', 'Mediums']
    wets.columns = ['Race', 'Date', 'Lap_Num', 'Tot_Laps', 'Wets']
    intr.columns = ['Race', 'Date', 'Lap_Num', 'Tot_Laps', 'Inters']

    race = pd.merge(race, hard, how = 'left', on = ['Race', 'Date', 'Lap_Num', 'Tot_Laps'])
    race = pd.merge(race, soft, how = 'left', on = ['Race', 'Date', 'Lap_Num', 'Tot_Laps'])
    race = pd.merge(race, medm, how = 'left', on = ['Race', 'Date', 'Lap_Num', 'Tot_Laps'])
    race = pd.merge(race, wets, how = 'left', on = ['Race', 'Date', 'Lap_Num', 'Tot_Laps'])
    race = pd.merge(race, intr, how = 'left', on = ['Race', 'Date', 'Lap_Num', 'Tot_Laps'])

    # Columns to fill NaN values with 0
    cols_fill = ['Hards', 'Softs', 'Mediums', 'Inters', 'Wets']

    # Fill NaN values with 0 in the specified columns
    race[cols_fill] = race[cols_fill].fillna(0)

    qf = df[['Race', 'Date', 'Driver', 'Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd']]
    qf = qf.drop_duplicates()
    qf = qf.loc[~qf['Qual_Sec1_Spd'].isna()].reset_index(drop = True)

    qual = qf.groupby(['Race', 'Date']).agg({'Qual_Sec1_Spd' : 'mean', 'Qual_Sec2_Spd' : 'mean', 'Qual_Sec3_Spd' : 'mean'}).reset_index()
    qual = qual.sort_values(by = 'Date').reset_index(drop = True)

    drql = qf.groupby(['Race', 'Date', 'Driver']).agg({'Qual_Sec1_Spd' : 'mean', 'Qual_Sec2_Spd' : 'mean', 'Qual_Sec3_Spd' : 'mean'}).reset_index()
    drql = drql.sort_values(by = ['Date', 'Driver']).reset_index(drop = True)
    drql.columns = ['Race', 'Date', 'Driver', 'Drv_Qual_Sec1', 'Drv_Qual_Sec2', 'Drv_Qual_Sec3']
    drql = pd.merge(drql, qual, on = ['Race', 'Date'], how = 'inner')

    drvr = pd.merge(drvr, drql, how = 'left', on = ['Race', 'Date', 'Driver'])
    drvr['Drv_Sec1'] = drvr['Drv_Sec1_Spd']/drvr['Qual_Sec1_Spd']
    drvr['Drv_Sec2'] = drvr['Drv_Sec2_Spd']/drvr['Qual_Sec2_Spd']
    drvr['Drv_Sec3'] = drvr['Drv_Sec3_Spd']/drvr['Qual_Sec3_Spd']

    drvr['DQ_Sec1'] = drvr['Drv_Qual_Sec1']/drvr['Qual_Sec1_Spd']
    drvr['DQ_Sec2'] = drvr['Drv_Qual_Sec2']/drvr['Qual_Sec2_Spd']
    drvr['DQ_Sec3'] = drvr['Drv_Qual_Sec3']/drvr['Qual_Sec3_Spd']

    drvr = drvr.drop(columns = ['Drv_Sec1_Spd', 'Drv_Sec2_Spd', 'Drv_Sec3_Spd', 'Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd', 'Drv_Qual_Sec1', 'Drv_Qual_Sec2', 'Drv_Qual_Sec3'])

    race = pd.merge(race, qual, how = 'inner', on = ['Race', 'Date'])

    race['Sec1'] = race['Race_Sec1_Spd']/race['Qual_Sec1_Spd']
    race['Sec2'] = race['Race_Sec2_Spd']/race['Qual_Sec2_Spd']
    race['Sec3'] = race['Race_Sec3_Spd']/race['Qual_Sec3_Spd']
    race['Progress'] = race['Lap_Num']/race['Tot_Laps']

    race['Status'] = 'S' + race['Track_Status'].astype(int).astype(str)

    columns_to_drop = [
        'Race_Sec1_Spd', 'Race_Sec2_Spd', 'Race_Sec3_Spd',
        'Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd',
        'Track_Status'
    ]

    race = race.drop(columns=columns_to_drop)

    race.columns = ['Race', 'Date', 'Lap', 'Total', 'Life', 'Drivers', 'Hards', 'Softs', 'Mediums', 'Inters', 'Wets', 'Sec1', 'Sec2', 'Sec3', 'Progress', 'Status']

    race = race[['Race', 'Date', 'Lap', 'Total', 'Progress', 'Status', 'Life', 'Drivers', 
                'Hards', 'Softs', 'Mediums', 'Inters', 'Wets', 'Sec1', 'Sec2', 'Sec3']]

    nos1 = race[['Race', 'Date', 'Lap', 'Total', 'Status']].loc[race['Status'] != 'S1'].reset_index(drop = True)
    nos1 = pd.get_dummies(nos1, columns = ['Status'])

    nos1[['Status_S2', 'Status_S4', 'Status_S6', 'Status_S7', 'Status_S8']] = nos1[['Status_S2', 'Status_S4', 'Status_S6', 'Status_S7', 'Status_S8']].astype(int)

    race = pd.merge(race, nos1, how = 'left', on = ['Race', 'Date', 'Lap', 'Total'])

    stat_cols = [col for col in race.columns if col.startswith('Status_')]

    # Fill NaN values with 0 for these columns
    race[stat_cols] = race[stat_cols].fillna(0)

    race = race.drop(columns = 'Status')

    race = race[['Race', 'Date', 'Lap', 'Total', 'Progress', 'Life', 'Drivers', 'Hards', 'Softs', 'Mediums', 'Inters',
                'Wets', 'Status_S2', 'Status_S4', 'Status_S6', 'Status_S7', 'Status_S8', 'Sec1', 'Sec2', 'Sec3']]

    drql['DQ_Sec1'] = drql['Drv_Qual_Sec1']/drql['Qual_Sec1_Spd']
    drql['DQ_Sec2'] = drql['Drv_Qual_Sec2']/drql['Qual_Sec2_Spd']
    drql['DQ_Sec3'] = drql['Drv_Qual_Sec3']/drql['Qual_Sec3_Spd']

    drql = drql.drop(columns = ['Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd', 'Drv_Qual_Sec1', 'Drv_Qual_Sec2', 'Drv_Qual_Sec3'])

    return race, qual, drql, drvr

def prev_rce_laps(group):
    records = []
    for index, row in group.iterrows():
        race, date, lap = row['Race'], row['Date'], row['Lap']
        # Check if all previous 10 laps exist
        previous_laps_indices = [
            group.index[group['Lap'] == lap - i].tolist()[0]
            for i in range(1, 11) if (lap - i) in group['Lap'].values
        ]
        # Only include the current lap if all previous laps exist
        if len(previous_laps_indices) == 10:
            #previous_laps_indices.append(index)
            records.append({'Index': index, 'Previous_Indices': sorted(previous_laps_indices)})
    return pd.DataFrame(records)

def process_laps(path, model, df, qual, nlaps):
    # Path to the checkpoint
    checkpoint_path = path

    # Load the saved model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # If there's a saved optimizer, you can load it as well
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    val_df = df.groupby(['Race', 'Date']).apply(prev_rce_laps).reset_index(drop=True).sort_values(by = 'Index').reset_index(drop = True)

    # 2. Create new columns in the DataFrame for 'Sec1_T', 'Sec2_T', and 'Sec3_T'
    df['Sec1_T'] = df['Sec1']
    df['Sec2_T'] = df['Sec2']
    df['Sec3_T'] = df['Sec3']

    # 3. Nullify 'Sec1', 'Sec2', and 'Sec3' if 'Total' - 'Lap' is less than or equal to 'nlaps'
    #df.loc[df['Total'] - df['Lap'] <= nlaps, ['Sec1', 'Sec2', 'Sec3']] = None
    df.loc[df['Lap'] > nlaps, ['Sec1', 'Sec2', 'Sec3']] = None

    mask = df.loc[val_df['Index'], 'Sec1'].isnull()

    # Filter val_df using the mask
    fvl_df = val_df[mask.values]  # Use .values to align the boolean mask with val_df's index
    fvl_df.reset_index(drop = True, inplace = True)

    for index, row in fvl_df.iterrows():
        input = []
        pred_row = df.iloc[row['Previous_Indices'], 4:20].values.astype('float32')
        input.append(pred_row)
        inp_tensor = torch.tensor(np.array(input))
        output = model(inp_tensor)
        v1 = output[:, 0].item()
        v2 = output[:, 1].item()
        v3 = output[:, 2].item()
        df.loc[row['Index'], 'Sec1'] = v1
        df.loc[row['Index'], 'Sec2'] = v2
        df.loc[row['Index'], 'Sec3'] = v3

    res = pd.merge(df, qual, how = 'inner', on = ['Race', 'Date'])

    res['Sec1'] = res['Sec1'] * res['Qual_Sec1_Spd']
    res['Sec2'] = res['Sec2'] * res['Qual_Sec2_Spd']
    res['Sec3'] = res['Sec3'] * res['Qual_Sec3_Spd']

    res['Sec1_T'] = res['Sec1_T'] * res['Qual_Sec1_Spd']
    res['Sec2_T'] = res['Sec2_T'] * res['Qual_Sec2_Spd']
    res['Sec3_T'] = res['Sec3_T'] * res['Qual_Sec3_Spd']

    res['Calc_Spd'] = (res['Sec1'] + res['Sec2'] + res['Sec3'])/3
    res['Real_Spd'] = (res['Sec1_T'] + res['Sec2_T'] + res['Sec3_T'])/3

    res = res.rename(columns={'Sec1': 'Sec1_P'})
    res = res.rename(columns={'Sec2': 'Sec2_P'})
    res = res.rename(columns={'Sec3': 'Sec3_P'})

    res['Err_Lap_Spd'] = ((res['Calc_Spd'] - res['Real_Spd']) / res['Real_Spd']) * 100

    return res

def preprocess_pos_model(tab, race, qual, drql, drvr):
    drv_race = pd.merge(tab, drvr, how = 'left', on = ['Race', 'Date', 'Lap', 'Total'])

    drv_race = pd.get_dummies(drv_race, columns=['Drv_Tyre'], prefix=['Drv_Tyre'])

    drv_race = drv_race[['Race', 'Date', 'Driver', 'Lap', 'Total', 'Sec1_T', 'Sec2_T', 'Sec3_T', 'Driver Points Before', 'Team Points Before', 'Grid Points Before', 'Progress', 'Drivers', 'Life', 'Sec1_P', 'Sec2_P', 'Sec3_P', 'Inlap', 'Outlap', 'Drv_Tyre_Life', 'Drv_Tyre_SOFT', 'Drv_Tyre_MEDIUM', 'Drv_Tyre_HARD', 'Drv_Tyre_INTERMEDIATE', 'Drv_Tyre_WET', 'Drv_Sec1', 'Drv_Sec2', 'Drv_Sec3', 'Position']]
    drv_race.columns = ['Race', 'Date', 'Driver', 'Lap', 'Total', 'Sec1_T', 'Sec2_T', 'Sec3_T', 'Driver Points Before', 'Team Points Before', 'Grid Points Before', 'Progress', 'Drivers', 'Life', 'Sec1_P', 'Sec2_P', 'Sec3_P', 'Inlap', 'Outlap', 'Drv_Tyre_Life', 'Soft', 'Medium', 'Hard', 'Inter', 'Wet', 'Drv_Sec1', 'Drv_Sec2', 'Drv_Sec3', 'Position']

    columns_to_convert = ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']

    # Convert boolean columns to integers
    drv_race[columns_to_convert] = drv_race[columns_to_convert].astype(int)

    drv_race = drv_race.dropna(subset=['Drv_Sec1', 'Position'])

    drv_race.reset_index(drop = True, inplace = True)

    drv_race = pd.merge(drv_race, drql, how = 'inner')

    drv_race = drv_race[['Race', 'Date', 'Driver', 'Lap', 'Total', 'Sec1_T', 'Sec2_T', 'Sec3_T', 'Driver Points Before', 'Team Points Before',
                        'Grid Points Before', 'Progress', 'Drivers', 'Life', 'Sec1_P', 'Sec2_P', 'Sec3_P', 'Inlap', 'Outlap', 'Drv_Tyre_Life',
                        'Soft', 'Medium', 'Hard', 'Inter', 'Wet', 'DQ_Sec1', 'DQ_Sec2', 'DQ_Sec3', 'Drv_Sec1', 'Drv_Sec2', 'Drv_Sec3', 'Position']] 
    #                     'Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd', 'Drv_Qual_Sec1', 'Drv_Qual_Sec2', 'Drv_Qual_Sec3', 

    drv_race['Race_Dist'] = np.where('Monaco' in drv_race['Race'], 260, 307)

    # Calculate lap length and sector length
    drv_race['Lap_Length'] = drv_race['Race_Dist'] / drv_race['Total']
    drv_race['Sector_Length'] = drv_race['Lap_Length'] / 3

    drv_race = pd.merge(drv_race, qual, how = 'inner', on = ['Race', 'Date'])


    # Compute sector times in seconds
    for sec in ['Sec1', 'Sec2', 'Sec3']:
        drv_race[f'Drv_{sec}_Time'] = (drv_race['Sector_Length'] / (drv_race[f'Drv_{sec}'] * drv_race[f'Qual_{sec}_Spd'])) * 60

    #drv_race = drv_race.drop(columns = ['Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd', 'Drv_Qual_Sec1', 'Drv_Qual_Sec2', 'Drv_Qual_Sec3'])


    # Calculate lap times as the sum of sector times
    drv_race['Laptime'] = drv_race[['Drv_Sec1_Time', 'Drv_Sec2_Time', 'Drv_Sec3_Time']].sum(axis=1)

    drv_race['Racetime'] = drv_race.groupby(['Date', 'Race', 'Driver']).apply(lambda x: x['Laptime'].cumsum()).reset_index(level=[0, 1, 2], drop=True)

    drv_race = drv_race.sort_values(by = ['Date', 'Lap', 'Driver'], ascending = [True, True, False]).reset_index(drop = True)

    return drv_race

def prev_drv_laps(group):
    records = []
    for index, row in group.iterrows():
        race, date, drv, lap = row['Race'], row['Date'], row['Driver'], row['Lap']
        # Check if all previous 10 laps exist
        previous_laps_indices = [
            group.index[group['Lap'] == lap - i].tolist()[0]
            for i in range(1, 11) if (lap - i) in group['Lap'].values
        ]
        # Only include the current lap if all previous laps exist
        if len(previous_laps_indices) == 10:
            #previous_laps_indices.append(index)
            records.append({'Index': index, 'Previous_Indices': sorted(previous_laps_indices)})
    return pd.DataFrame(records)

def driver_laps(path, model, nf, nlaps):
    # Path to the checkpoint
    checkpoint_path = path

    # Load the saved model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # If there's a saved optimizer, you can load it as well
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    df = nf.copy()

    df['Sec1_P'] = df['Sec1_P']/df['Qual_Sec1_Spd']
    df['Sec2_P'] = df['Sec2_P']/df['Qual_Sec2_Spd']
    df['Sec3_P'] = df['Sec3_P']/df['Qual_Sec3_Spd']

    df['Sec1_T'] = df['Sec1_T']/df['Qual_Sec1_Spd']
    df['Sec2_T'] = df['Sec2_T']/df['Qual_Sec2_Spd']
    df['Sec3_T'] = df['Sec3_T']/df['Qual_Sec3_Spd']


    val_df = df.groupby(['Race', 'Date', 'Driver']).apply(prev_drv_laps).reset_index(drop=True).sort_values(by = 'Index').reset_index(drop = True)

    df['Drv_Sec1_T'] = df['Drv_Sec1'].copy()
    df['Drv_Sec2_T'] = df['Drv_Sec2'].copy()
    df['Drv_Sec3_T'] = df['Drv_Sec3'].copy()

    df['Drv_Sec1_Time_T'] = df['Drv_Sec1_Time'].copy()
    df['Drv_Sec2_Time_T'] = df['Drv_Sec2_Time'].copy()
    df['Drv_Sec3_Time_T'] = df['Drv_Sec3_Time'].copy()

    df['Laptime_T'] = df['Laptime'].copy()
    #df['Racetime_T'] = df['Racetime'].copy()
    df.drop(['Position', 'Racetime'], axis=1, inplace=True)
    
    # 3. Nullify 'Sec1', 'Sec2', and 'Sec3' if 'Total' - 'Lap' is less than or equal to 'nlaps'
    #df.loc[df['Total'] - df['Lap'] <= nlaps, ['Sec1', 'Sec2', 'Sec3']] = None
    df.loc[df['Lap'] > nlaps, ['Drv_Sec1', 'Drv_Sec2', 'Drv_Sec3', 'Drv_Sec1_Time', 'Drv_Sec2_Time', 'Drv_Sec3_Time', 'Laptime', 'Racetime' ]] = None

    mask = df.loc[val_df['Index'], 'Drv_Sec1'].isnull()

    # Filter val_df using the mask
    fvl_df = val_df[mask.values]  # Use .values to align the boolean mask with val_df's index
    fvl_df.reset_index(drop = True, inplace = True)
    fvl_df['Lap'] = df.loc[fvl_df['Index'], 'Lap'].values
    fvl_df['Date'] = df.loc[fvl_df['Index'], 'Date'].values
    fvl_df['Race'] = df.loc[fvl_df['Index'], 'Race'].values

    prev_race = fvl_df.loc[0, 'Race']
    prev_date = fvl_df.loc[0, 'Date']
    prev_lap = fvl_df.loc[0, 'Lap']

    for index, row in fvl_df.iterrows():
        input = []
        curr_race = row['Race']
        curr_date = row['Date']
        curr_lap = row['Lap']

        prev_race = curr_race
        prev_date = curr_date
        prev_lap = curr_lap
        pred_row = df.iloc[row['Previous_Indices'], 8:31].values.astype('float32')
        input.append(pred_row)
        inp_tensor = torch.tensor(np.array(input))
        output = model(inp_tensor)
        v1 = output[:, 0].item()
        v2 = output[:, 1].item()
        v3 = output[:, 2].item()

        df.loc[row['Index'], 'Drv_Sec1'] = v1
        df.loc[row['Index'], 'Drv_Sec2'] = v2
        df.loc[row['Index'], 'Drv_Sec3'] = v3

        # Compute sector times in seconds
        for sec in ['Sec1', 'Sec2', 'Sec3']:
            df.loc[row['Index'], f'Drv_{sec}_Time'] = (df.loc[row['Index'], 'Sector_Length'] / (df.loc[row['Index'], f'Drv_{sec}'] * df.loc[row['Index'], f'Qual_{sec}_Spd'])) * 60

        df.loc[row['Index'], 'Laptime'] = df.loc[row['Index'], ['Drv_Sec1_Time', 'Drv_Sec2_Time', 'Drv_Sec3_Time']].sum()

    res = df.copy()

    res['Drv_Sec1'] = res['Drv_Sec1'] * res['Qual_Sec1_Spd']
    res['Drv_Sec2'] = res['Drv_Sec2'] * res['Qual_Sec2_Spd']
    res['Drv_Sec3'] = res['Drv_Sec3'] * res['Qual_Sec3_Spd']

    res['Drv_Sec1_T'] = res['Drv_Sec1_T'] * res['Qual_Sec1_Spd']
    res['Drv_Sec2_T'] = res['Drv_Sec2_T'] * res['Qual_Sec2_Spd']
    res['Drv_Sec3_T'] = res['Drv_Sec3_T'] * res['Qual_Sec3_Spd']

    res['Drv_Calc_Spd'] = (res['Drv_Sec1'] + res['Drv_Sec2'] + res['Drv_Sec3'])/3
    res['Drv_Real_Spd'] = (res['Drv_Sec1_T'] + res['Drv_Sec2_T'] + res['Drv_Sec3_T'])/3

    res = res.rename(columns={'Drv_Sec1': 'Drv_Sec1_P'})
    res = res.rename(columns={'Drv_Sec2': 'Drv_Sec2_P'})
    res = res.rename(columns={'Drv_Sec3': 'Drv_Sec3_P'})
    
    res = res.rename(columns={'Drv_Sec1_Time': 'Drv_Sec1_Time_P'})
    res = res.rename(columns={'Drv_Sec2_Time': 'Drv_Sec2_Time_P'})
    res = res.rename(columns={'Drv_Sec3_Time': 'Drv_Sec3_Time_P'})

    res = res.rename(columns={'Laptime': 'Laptime_P'})
    
    #res['Laptime_P'] = res['Drv_Sec1_Time_P'] + res['Drv_Sec2_Time_P'] + res['Drv_Sec3_Time_P']

    #res = res.rename(columns={'Racetime': 'Racetime_P'})

    res = res.sort_values(['Date', 'Driver', 'Lap'])
    res = res.drop_duplicates()
    #res.drop(columns=['Racetime_T', 'Racetime_P'], inplace=True)

    res['Racetime_P'] = res.groupby(['Date', 'Driver'])['Laptime_P'].cumsum()
    res['Racetime_T'] = res.groupby(['Date', 'Driver'])['Laptime_T'].cumsum()

    rpos = res.groupby(['Date', 'Race', 'Lap']).agg({'Racetime_P': ['mean', 'std']}).reset_index()
    rpos.columns = ['Date', 'Race', 'Lap', 'Race_P_Mean', 'Race_P_Std']

    res = pd.merge(res, rpos, how = 'inner', on = ['Date', 'Race', 'Lap'])
    res['Position_P'] = (res['Race_P_Mean'] - res['Racetime_P']) / (res['Race_P_Std'])

    res.drop(columns=['Race_P_Mean', 'Race_P_Std'], inplace=True)

    rpos = res.groupby(['Date', 'Race', 'Lap']).agg({'Racetime_T': ['mean', 'std']}).reset_index()
    rpos.columns = ['Date', 'Race', 'Lap', 'Race_T_Mean', 'Race_T_Std']
    
    res = pd.merge(res, rpos, how = 'inner', on = ['Date', 'Race', 'Lap'])
    res['Position_T'] = (res['Race_T_Mean'] - res['Racetime_T']) / (res['Race_T_Std'])

    res = res.rename(columns={'Drv_Tyre_Life': 'Tire_Life'})

    res['Tire'] = res[['Soft', 'Medium', 'Hard', 'Inter', 'Wet']].apply(lambda row: ', '.join(row.index[row == 1]), axis=1)

    res.drop(columns=['Race_T_Mean', 'Race_T_Std', 'Driver Points Before', 'Team Points Before', 'Grid Points Before',
                      'Progress', 'Drivers', 'Sec1_T', 'Sec2_T', 'Sec3_T', 'Sec1_P', 'Sec2_P', 'Sec3_P', 'Life',
                      'DQ_Sec1', 'DQ_Sec2', 'DQ_Sec3', 'Race_Dist', 'Sector_Length', 'Lap_Length', 'Drv_Sec1_Time_P',
                      'Drv_Sec2_Time_P', 'Drv_Sec3_Time_P', 'Drv_Sec1_Time_T', 'Drv_Sec2_Time_T', 'Drv_Sec3_Time_T', 
                      'Soft', 'Medium', 'Hard', 'Inter', 'Wet', 'Racetime'
                      ], inplace=True)
    
    res['Inlap'] = res['Inlap'].astype(int)
    res['Outlap'] = res['Outlap'].astype(int)

    res['Inlap'].replace({0: 'N', 1: 'Y'}, inplace=True)
    res['Outlap'].replace({0: 'N', 1: 'Y'}, inplace=True)

    
    res = res[['Race', 'Date', 'Driver', 'Lap', 'Total', 'Tire', 'Tire_Life', 'Qual_Sec1_Spd', 'Qual_Sec2_Spd', 'Qual_Sec3_Spd',
               'Inlap', 'Outlap', 'Drv_Sec1_T', 'Drv_Sec2_T', 'Drv_Sec3_T', 'Laptime_T', 'Racetime_T', 'Position_T', 'Drv_Real_Spd',
               'Drv_Sec1_P', 'Drv_Sec2_P', 'Drv_Sec3_P', 'Laptime_P', 'Racetime_P', 'Position_P', 'Drv_Calc_Spd'
               ]]

    res['Err_Drv_Spd'] = ((res['Drv_Calc_Spd'] - res['Drv_Real_Spd']) / res['Drv_Real_Spd']) * 100
    res['Err_Position'] = res['Position_P'] - res['Position_T']

    return res

def rank_drivers(df):
    prg = df.copy()
    # Calculate the number of laps completed and the total time taken by each driver in each race
    driver_stats = prg.groupby(['Race', 'Driver']).agg(
        Laps_completed=('Lap', 'max'),
        Finish_T=('Racetime_T', 'max')
    ).reset_index()

    # Sort drivers for each race by laps completed (descending) and total time taken (ascending)
    driver_stats = driver_stats.sort_values(
        ['Race', 'Laps_completed', 'Finish_T'],
        ascending=[True, False, True]
    )

    # Assign standings within each race
    driver_stats['Standing_T'] = driver_stats.groupby('Race').cumcount() + 1

    # Merge the 'Finish_T' and 'Standing' back into the original dataframe
    prg = prg.merge(driver_stats[['Race', 'Driver', 'Finish_T', 'Standing_T']], on=['Race', 'Driver'], how='left')

    driver_stats = prg.groupby(['Race', 'Driver']).agg(
        Laps_completed=('Lap', 'max'),
        Finish_P=('Racetime_P', 'max')
    ).reset_index()

    # Sort drivers for each race by laps completed (descending) and total time taken (ascending)
    driver_stats = driver_stats.sort_values(
        ['Race', 'Laps_completed', 'Finish_P'],
        ascending=[True, False, True]
    )

    # Assign standings within each race
    driver_stats['Standing_P'] = driver_stats.groupby('Race').cumcount() + 1

    # Merge the 'Finish_T' and 'Standing' back into the original dataframe
    prg = prg.merge(driver_stats[['Race', 'Driver', 'Finish_P', 'Standing_P']], on=['Race', 'Driver'], how='left')

    prg.drop(columns = ['Finish_T', 'Finish_P'], inplace = True)

    std = prg.groupby(['Race', 'Date', 'Driver', 'Total']).agg({'Lap' : 'max', 'Standing_T' : 'mean', 'Standing_P' : 'mean'}).reset_index()

    std.columns = ['Race', 'Date', 'Driver', 'Total', 'Laps Completed', 'Standing_T', 'Standing_P']
    return std