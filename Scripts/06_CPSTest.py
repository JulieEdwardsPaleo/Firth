import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import netCDF4 as nc
from datetime import datetime
import utils as u  
from statsmodels.stats.stattools import durbin_watson

directory_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronologies')

combined_df = pd.DataFrame()

for filename in os.listdir(directory_path):
    if filename.startswith("noFill_"):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        std_series = df.set_index('Unnamed: 0')['std']
        column_name = filename.split('noFill_')[1].split('.ind')[0]
        combined_df[column_name] = std_series

sfrcs = combined_df

file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Climate', 'iera5_t2m_daily_-141.05E_68.67N_n.nc')

dataset = nc.Dataset(file_path)

t2m = dataset.variables['t2m'][:]
time = dataset.variables['time'][:]
time_units = dataset.variables['time'].units
dates = nc.num2date(time, time_units)
dates = [pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates]
df = pd.DataFrame({'date': dates, 'temperature': t2m})
daily = df
daily['year'] = daily['date'].dt.year
daily['month'] = daily['date'].dt.month
daily['doy'] = daily['date'].dt.dayofyear

# Filter data for July 1 to August 31 period and calculate average temperature per year
july_1_doy = 182  # Day of year for July 1, 182
aug_31_doy = 243   # Day of year for August 31, 243
filtered_df = daily[(daily['doy'] >= july_1_doy) & (daily['doy'] <= aug_31_doy)]
avg_temp_per_year = filtered_df.groupby('year')['temperature'].mean()
data = pd.merge(avg_temp_per_year, sfrcs, left_index=True, right_index=True)
data.dropna(subset=['temperature'], inplace=True)
y = data['temperature']



# Calibration and validration periods
cal1 = data.index[:36]
val1 = data.index[36:]
cal2 = data.index[36:]
val2 = data.index[:36]

proxy_columns = data.columns.drop('temperature')

results = {
    'Proxy': [],
    'R2 (calibration) - Set1': [],
    'R2 (validation) - Set1': [],
    'Reduction of Error - Set1': [],
    'Coefficient of Efficiency - Set1': [],
    'Durbin-Watson - Set1': [],
    'R2 (calibration) - Set2': [],
        'R2 (validation) - Set2': [],
    'Reduction of Error - Set2': [],
    'Coefficient of Efficiency - Set2': [],
    'Durbin-Watson - Set2': [],
    'R2 (calibration) - full': [],
    'R2 (validation) - full': [],
    'Reduction of Error - full': [],
    'Coefficient of Efficiency - full': [],
    'Durbin-Watson - full': []
}

for proxy_column in proxy_columns:
    proxy = data[[proxy_column]]
    scaler = StandardScaler()
    proxy_scaled = scaler.fit_transform(proxy)

    # Run CPS analysis for Early cal/Late Val
    yhat1, s3R2v1, s3R2c1, s3RE1, s3CE1 = u.simple_cps(y, y.index, proxy_scaled, data.index, cal1, val1)
    resid1 = y - yhat1
    db1 = durbin_watson(resid1)

    # Run CPS analysis for Late cal/early val
    yhat2, s3R2v2, s3R2c2, s3RE2, s3CE2 = u.simple_cps(y, y.index, proxy_scaled, data.index, cal2, val2)
    resid2 = y - yhat2
    db2 = durbin_watson(resid2)

    # full cal/val period
    yhatf, s3R2vf, s3R2cf, s3REf, s3CEf = u.simple_cps(y, y.index, proxy_scaled, data.index, y.index, y.index)
    residfull = y - yhatf
    dbf = durbin_watson(residfull)

    results['Proxy'].append(proxy_column)
    results['R2 (calibration) - Set1'].append(s3R2c1)
    results['R2 (validation) - Set1'].append(s3R2v1)
    results['Reduction of Error - Set1'].append(s3RE1)
    results['Coefficient of Efficiency - Set1'].append(s3CE1)
    results['Durbin-Watson - Set1'].append(db1)
    results['R2 (validation) - Set2'].append(s3R2v2)
    results['R2 (calibration) - Set2'].append(s3R2c2)
    results['Reduction of Error - Set2'].append(s3RE2)
    results['Coefficient of Efficiency - Set2'].append(s3CE2)
    results['Durbin-Watson - Set2'].append(db2)
    results['R2 (validation) - full'].append(s3R2vf)
    results['R2 (calibration) - full'].append(s3R2cf)
    results['Reduction of Error - full'].append(s3REf)
    results['Coefficient of Efficiency - full'].append(s3CEf)
    results['Durbin-Watson - full'].append(dbf)


results_df = pd.DataFrame(results)
w1, w2, w3, w4, w5 = 1, 1, 1, 1, 2  # Heavier weight for DW statistic

results_df['Score'] = (w1 * results_df['R2 (validation) - Set1'] +
                       w2 * results_df['R2 (calibration) - Set1'] +
                       w3 * results_df['Reduction of Error - Set1'] +
                       w4 * results_df['Coefficient of Efficiency - Set1'] -
                       w5 * abs(results_df['Durbin-Watson - Set1'] - 2) +
                       w1 * results_df['R2 (validation) - Set2'] +
                       w2 * results_df['R2 (calibration) - Set2'] +
                       w3 * results_df['Reduction of Error - Set2'] +
                       w4 * results_df['Coefficient of Efficiency - Set2'] -
                       w5 * abs(results_df['Durbin-Watson - Set2'] - 2) +
                       w1 * results_df['R2 (validation) - full'] +
                       w2 * results_df['R2 (calibration) - full'] +
                       w3 * results_df['Reduction of Error - full'] +
                       w4 * results_df['Coefficient of Efficiency - full'] -
                       w5 * abs(results_df['Durbin-Watson - full'] - 2))
results_df_sorted = results_df.sort_values(by='Score', ascending=False)

print(results_df_sorted)
best_row = results_df_sorted.iloc[0:10]

results_df_sorted.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronologies','Scored_Bestproxy_results_.csv'), index=False)