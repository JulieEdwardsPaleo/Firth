import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import netCDF4 as nc
from datetime import datetime
import utils as u  
from statsmodels.stats.stattools import durbin_watson

# Directory containing the MXD data files
directory_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronologies')

# Initialize an empty dataframe for concatenation
combined_df = pd.DataFrame()

# Loop through all files in the directory and process those starting with 'noFill_'
for filename in os.listdir(directory_path):
    if filename.startswith("noFill_"):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        std_series = df.set_index('Unnamed: 0')['std']
        column_name = filename.split('noFill_')[1].split('.ind')[0]
        combined_df[column_name] = std_series

# Assign the combined dataframe to sfrcs
sfrcs = combined_df

# Load the NetCDF dataset
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Climate', 'iera5_t2m_daily_-141.05E_68.67N_n.nc')

dataset = nc.Dataset(file_path)

# Extract temperature and time variables
t2m = dataset.variables['t2m'][:]
time = dataset.variables['time'][:]
time_units = dataset.variables['time'].units
dates = nc.num2date(time, time_units)
dates = [pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates]

# Create a DataFrame for daily temperature data
df = pd.DataFrame({'date': dates, 'temperature': t2m})
daily = df
daily['year'] = daily['date'].dt.year
daily['month'] = daily['date'].dt.month
daily['doy'] = daily['date'].dt.dayofyear

# Filter data for July 1 to August 31 period and calculate average temperature per year
july_1_doy = 182  # Day of year for July 1
aug_31_doy = 243   # Day of year for August 31
filtered_df = daily[(daily['doy'] >= july_1_doy) & (daily['doy'] <= aug_31_doy)]
avg_temp_per_year = filtered_df.groupby('year')['temperature'].mean()


# Merge the average temperature data with the proxy data
data = pd.merge(avg_temp_per_year, sfrcs, left_index=True, right_index=True)
data.dropna(subset=['temperature'], inplace=True)
y = data['temperature']

# Split data into calibration and validation sets
cal1 = data.index[:36]
val1 = data.index[36:]
cal2 = data.index[36:]
val2 = data.index[:36]

# Identify proxy columns
proxy_columns = data.columns.drop('temperature')

# Initialize results dictionary for both sets
results = {
    'Proxy': [],
    'R2 (validation) - Set1': [],
    'R2 (calibration) - Set1': [],
    'Reduction of Error - Set1': [],
    'Coefficient of Efficiency - Set1': [],
    'Durbin-Watson - Set1': [],
    'R2 (validation) - Set2': [],
    'R2 (calibration) - Set2': [],
    'Reduction of Error - Set2': [],
    'Coefficient of Efficiency - Set2': [],
    'Durbin-Watson - Set2': [],
    'R2 (validation) - full': [],
    'R2 (calibration) - full': [],
    'Reduction of Error - full': [],
    'Coefficient of Efficiency - full': [],
    'Durbin-Watson - full': []
}

# Loop through each proxy column
for proxy_column in proxy_columns:
    proxy = data[[proxy_column]]
    scaler = StandardScaler()
    proxy_scaled = scaler.fit_transform(proxy)

    # Run CPS analysis for Set 1
    yhat1, s3R2v1, s3R2c1, s3RE1, s3CE1 = u.simple_cps(y, y.index, proxy_scaled, data.index, cal1, val1)
    resid1 = y - yhat1
    db1 = durbin_watson(resid1)

    # Run CPS analysis for Set 2
    yhat2, s3R2v2, s3R2c2, s3RE2, s3CE2 = u.simple_cps(y, y.index, proxy_scaled, data.index, cal2, val2)
    resid2 = y - yhat2
    db2 = durbin_watson(resid2)

    yhatf, s3R2vf, s3R2cf, s3REf, s3CEf = u.simple_cps(y, y.index, proxy_scaled, data.index, y.index, y.index)
    residfull = y - yhatf
    dbf = durbin_watson(residfull)

    # Store results in dictionary
    results['Proxy'].append(proxy_column)
    results['R2 (validation) - Set1'].append(s3R2v1)
    results['R2 (calibration) - Set1'].append(s3R2c1)
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

# Calculate the score for each row
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
# Sort by the score
results_df_sorted = results_df.sort_values(by='Score', ascending=False)

# Display the row with the highest score
best_row = results_df_sorted.iloc[0:10]

best_row.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronologies','Scored_Bestproxy_results_CRUJA.csv'), index=False)