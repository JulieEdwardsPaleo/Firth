
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
import powerlaw
import pymannkendall as mk
import netCDF4 as nc
from datetime import datetime
import os
import xarray as xr
import multitaper.mtspec as mtspec
from csaps import csaps
from math import cos, pi
from scipy.signal import lfilter
from sklearn.preprocessing import scale


models=pd.read_csv('/Users/julieedwards/Documents/Projects/MANCHA/Climate/models/jul_aug_avg_temperature.csv',index_col='year')


start_year = 1150
end_year = 2021
spectra = []

nw = 4
kspec = int(2 * nw - 1)

def interpolate_nans(data):
    nans, x = np.isnan(data), lambda z: z.nonzero()[0]
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data
for col in models.columns:
    model_data = models[col].loc[start_year:end_year].values
    model_data = interpolate_nans(model_data)  # Interpolate over NaNs
    spec = mtspec.MTSpec(model_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None)
    spectra.append(spec.rspec()[1])

# Convert spectra list to a numpy array
spectra = np.array(spectra)

# Calculate the 5% and 95% percentiles
lower_bound = np.percentile(spectra, 5, axis=0)
upper_bound = np.percentile(spectra, 95, axis=0)
median = np.percentile(spectra, 50, axis=0)

# Get the periods
periods = 1 / spec.rspec()[0]



def long_cps(climate,cyr,proxy,pyr,calibrate,validate):
    climate.index=cyr
    icalc = np.intersect1d(cyr,calibrate)
    icalp=np.intersect1d(pyr,calibrate)
    ivalc = np.intersect1d(cyr, validate)
    ivalp=np.intersect1d(pyr,validate)
    yhat= np.empty((len(proxy),1))
    if proxy.ndim >1:
        proxy = pd.Series(np.mean(scale(proxy),axis=1))
    else:
       proxy=pd.Series(scale(proxy))
    proxy.index=pyr
    mu = np.mean(climate[icalc])
    sig = np.std(climate[icalc],ddof=1)
    xm=np.mean(proxy[icalp])
    xs=np.std(proxy[icalp],ddof=1)
    yhat= ((proxy-xm)/xs)*sig + mu
    s3ec = yhat[icalp] - climate[icalc]; # calibration period residuals
    s3ev = yhat[ivalp] - climate[ivalc]; # validation period residuals
    cbar = np.mean(climate[icalc])
    s3RE= 1 - np.sum((climate[ivalc] - yhat[ivalp]) ** 2) / np.nansum((climate[ivalc] - cbar) ** 2)
    vbar = np.mean(climate[ivalc])
    s3CE= 1 - np.sum((climate[ivalc] - yhat[ivalp]) ** 2) / np.nansum((climate[ivalc] - vbar) ** 2)
    rhoc = np.corrcoef(climate[icalc],yhat[icalp])
    RSS = np.sum((climate[icalc] - yhat[icalc]) ** 2)
    TSS = np.sum((climate[icalc] - climate[icalc].mean()) ** 2)
    CE = 1 - (RSS / TSS)

    #s3rhov = rhov[0,1]
   # s3R2v  = rhov[0,1] ** 2 # square the off-diagonal element
    s3R2c  = rhoc[0,1] ** 2 # square the off-diagonal element
    return yhat, s3R2c,s3RE, s3CE

directory_path = "/Users/julieedwards/Documents/Projects/MANCHA/MXD/nomcrb08_June2024/"
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

file_path = '/Users/julieedwards/Documents/Projects/MANCHA/Climate/daily/iera5_t2m_daily_-141.05E_68.67N_n.nc'
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

import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# Assuming long_cps is a function already defined somewhere in your code
# Also assuming y is a pd.Series containing the average temperature per year in Kelvin

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

    yhat, s3R2c, s3RE, s3CE = long_cps(y, y.index, proxy_scaled, sfrcs.index, y.index, y.index)
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

f_path = '/Users/julieedwards/Documents/Projects/MANCHA/MXD/FR_MXD.csv'

og=pd.read_csv(f_path)
og.index=og['age_AD']
ogoverlap=og.loc[1150:2002]['trsgiFir']

ogyhat, ogs3R2c, ogs3RE, ogs3CE = long_cps(avg_temp_per_year-273.15, avg_temp_per_year.index, ogoverlap, ogoverlap.index, avg_temp_per_year.loc[1950:2002].index, avg_temp_per_year.loc[1950:2002].index)


sfrcs=recons_df[['rcspbw10','rcspbw20','rcspbw40','rcspbw80']]

lower_bound = lower_bound.flatten()
upper_bound = upper_bound.flatten()
median = median.flatten()
periods = periods.flatten()
start_year = 1150
end_year = 1800
colors=['#EB72AF','#B6306B','#84143D','#52000F']
lab=['aMXD 10 $\mu$m','aMXD 20 $\mu$m','aMXD 40 $\mu$m','aMXD 80 $\mu$m']

fig, axs = plt.subplots(1,2,figsize=(6, 3),sharey=True,sharex=True)

for i, column in enumerate(sfrcs.columns):
            #sfrcs_data = spline(np.arange(Start,End+1),sfrcs[(sfrcs.index >= Start) & (sfrcs.index <= End)][f'{prefix}{res}'].values,2)
            #rcs_data = spline(np.arange(Start,End+1),rcs[(rcs.index >= Start) & (rcs.index <= End)][f'{prefix}{res}'].values,2)
    sfrcs_data=sfrcs.loc[start_year:end_year][column].values
    x_sfrcs = mtspec.MTSpec(sfrcs_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()

    periods_sfrcs = 1 / x_sfrcs[0]
    #ax = axs[i]
    #mod=ax.fill_between(periods, lower_bound, upper_bound, color='lightgray', label='5%-95% envelope')
    axs[0].plot(periods_sfrcs, x_sfrcs[1], label=lab[i],color=colors[i])
og_data=ogyhat.loc[start_year:end_year].values
x_og = mtspec.MTSpec(og_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()
periods_og = 1 / x_og[0]
#axs[0].plot(periods_og, x_og[1], label='2013MXD',color='k')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid(True,linestyle='--')
axs[0].legend(frameon=False,fontsize=9,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left')

axs[0].set_xlabel('Log(Period (years))',fontsize=9)
axs[0].set_ylabel('Log(PSD)')
axs[0].set_title('Pre-industrial aMXD spectra', y=1,fontsize=9)
start_year = 1150
end_year = 2021
for i, column in enumerate(sfrcs.columns):
            #sfrcs_data = spline(np.arange(Start,End+1),sfrcs[(sfrcs.index >= Start) & (sfrcs.index <= End)][f'{prefix}{res}'].values,2)
            #rcs_data = spline(np.arange(Start,End+1),rcs[(rcs.index >= Start) & (rcs.index <= End)][f'{prefix}{res}'].values,2)
    sfrcs_data=sfrcs.loc[start_year:end_year][column].values
    x_sfrcs = mtspec.MTSpec(sfrcs_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()

    periods_sfrcs = 1 / x_sfrcs[0]
    #ax = axs[i]
    #mod=ax.fill_between(periods, lower_bound, upper_bound, color='lightgray', label='5%-95% envelope')
    axs[1].plot(periods_sfrcs, x_sfrcs[1], label=lab[i],color=colors[i])
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid(True,linestyle='--')
axs[1].legend(frameon=False,fontsize=9,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left')
axs[1].set_xlabel('Log(Period (years))',fontsize=9)
axs[1].set_title('Full period aMXD spectra', y=1,fontsize=9)
axs[0].text(.7,250,'a)')
axs[1].text(.7,250,'b)')

plt.tight_layout()
plt.savefig('spectraMTM.eps',format='eps',bbox_inches='tight')
plt.show()
