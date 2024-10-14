import pandas as pd
from csaps import csaps
import numpy as np
from math import cos, pi
from sklearn.preprocessing import StandardScaler
import os
import netCDF4 as nc
from datetime import datetime
import utils as u  
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import acf
import matplotlib.gridspec as gridspec

## data read in of aMXD chronologies

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
sfrcs=sfrcs.loc[1150:2021]


## Data read in of ERA5 data
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Climate', 'iera5_t2m_daily_-141.05E_68.67N_n.nc')
dataset = nc.Dataset(file_path)
t2m = dataset.variables['t2m'][:]
time = dataset.variables['time'][:]
time_units = dataset.variables['time'].units
dates = nc.num2date(time, time_units)
dates = [pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates]
df = pd.DataFrame({'date': dates, 'temperature': t2m})
df = df[(df['date'].dt.year >= 1950) & (df['date'].dt.year <= 2021)]
daily = df
daily['year'] = daily['date'].dt.year
daily['month'] = daily['date'].dt.month
daily['doy'] = daily['date'].dt.dayofyear
july_doy = 195  # 
aug_doy = 223   # 
filtered_df = daily[(daily['doy'] >= july_doy) & (daily['doy'] <= aug_doy)]
avg_temp_per_year = filtered_df.groupby('year')['temperature'].mean()

y=avg_temp_per_year-273.15


# set up results_df which will have the reconstruction statistics
proxy_columns = sfrcs.columns
results = {
    'Proxy': [],
    'R2 (calibration)': [],
    'Reduction of Error': [],
    'Coefficient of Efficiency': [],
    'Durbin-Watson': [],
    'RMSE': [],
    'Standard Error': []}

Recons = {}
for proxy_column in proxy_columns:
    proxy = sfrcs[[proxy_column]]
    scaler = StandardScaler()
    proxy_scaled = scaler.fit_transform(proxy)
# reconstructions using aMXD and calibrated to the temperature for DOY 195-223
    yhat, s3R2c, s3RE, s3CE = u.long_cps(y, y.index, proxy_scaled, sfrcs.index, y.index, y.index)
    residfull = y - yhat
    dbf = durbin_watson(residfull.dropna())
    rmse = np.sqrt(np.mean(residfull ** 2))
    std_error = np.std(residfull) / np.sqrt(len(residfull))
    results['Proxy'].append(proxy_column)
    results['R2 (calibration)'].append(s3R2c)
    results['Reduction of Error'].append(s3RE)
    results['Coefficient of Efficiency'].append(s3CE)
    results['Durbin-Watson'].append(dbf)
    results['RMSE'].append(rmse)
    results['Standard Error'].append(std_error)
    Recons[proxy_column] = yhat

results_df = pd.DataFrame(results)
recons_df = pd.DataFrame(Recons)
sfrcs_p = recons_df[['pbw10','pbw20','pbw40','pbw80','pbw100']]



# read in Anchukaitis et al., 2013 MXD data (from ITRDB)
f_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'FR_MXD.csv')
og=pd.read_csv(f_path)
og.index=og['age_AD']
ogoverlap=og.loc[1150:2002]['trsgiFir']

ogyhat, ogs3R2c, ogs3RE, ogs3CE = u.long_cps(avg_temp_per_year-273.15, avg_temp_per_year.index, ogoverlap, ogoverlap.index, avg_temp_per_year.loc[1950:2002].index, avg_temp_per_year.loc[1950:2002].index)

## plotting
ogr=[]
ogbwr=[]
for column in sfrcs_p.columns:
    r=pearsonr(ogoverlap,sfrcs_p.loc[1150:2002][column])
    ogr.append(r.statistic)
for column in sfrcs_p.columns:
    rbw=pearsonr(ogoverlap,sfrcs_p.loc[1150:2002][column])
    ogbwr.append(rbw.statistic)
fig=plt.subplots(figsize=(3.5,3))
plt.plot(ogbwr,'-o',color='k')
plt.plot(ogr,':o',color='k')
plt.ylim(0.75,0.9)
plt.xticks([0,1,2,3,4],labels=['aMXD10','aMXD20','aMXD40','aMXD80','aMXD100'],fontsize=9)
plt.grid()
plt.title('Correlation with 2013MXD')
plt.ylabel('R')
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', '2013corr.eps'), format='eps',bbox_inches='tight')





def get_param(amp, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) /(12 * (cos(2 * pi * freq) - 1) ** 2))+ 1)
    return spline_param

# Function to fit a spline curve to the series
def spline(x, y, period=None):
    if period is None:
        period = len(x) * 0.67
    
    p = get_param(0.5, period)
    yi = csaps(x, y, x, smooth=p)
    return yi


## plotting 2013 MXD with aMXD

splineamount=100
fig=plt.figure(figsize=(6,3))
plt.plot(ogoverlap.index,spline(ogyhat.index,ogyhat,100),'k',linewidth=2,label='2013MXD')
plt.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw10'],splineamount),label='aMXD 10 $\mu$m',color='#D75E9B')
plt.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw20'],splineamount),label='aMXD 20 $\mu$m',color='#A21C57')
plt.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw80'],splineamount),label='aMXD 80 $\mu$m',color='#660C23')
plt.legend(frameon=False,fontsize=9)
plt.grid(linewidth=0.5)
plt.ylim(6,12)
plt.xlim(1150,2021)
plt.xlabel('Year',fontsize=9)
plt.ylabel('Temperature ($^\circ$C)',fontsize=9)
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', '2013MXDaMXD.eps'), format='eps',bbox_inches='tight')
