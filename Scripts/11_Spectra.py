
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
import powerlaw
import netCDF4 as nc
from datetime import datetime
import os
import xarray as xr
import multitaper.mtspec as mtspec
from csaps import csaps
from math import cos, pi
from scipy.signal import lfilter
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson





start_year = 1150
end_year = 2021
spectra = []

nw = 4
kspec = int(2 * nw - 1)




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


sfrcs=recons_df[['pbw10','pbw20','pbw80']]


start_year = 1150
end_year = 1800
colors=['#D75E6A','#A21C57','#49006a']
lab=['aMXD 10 $\mu$m','aMXD 20 $\mu$m','aMXD 80 $\mu$m']

fig, axs = plt.subplots(1,3,figsize=(6.5, 3),sharey=True,sharex=False)

for i, column in enumerate(sfrcs.columns):
    sfrcs_data=sfrcs.loc[start_year:end_year][column].values
    x_sfrcs = mtspec.MTSpec(sfrcs_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()

    periods_sfrcs = 1 / x_sfrcs[0]
    axs[0].plot(periods_sfrcs, x_sfrcs[1], label=lab[i],color=colors[i])

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid(True,linestyle='--',color=[0.8,0.8,0.8])
axs[0].legend(frameon=False,fontsize=9,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left')

axs[0].set_xlabel('Log(Period (years))',fontsize=9)
axs[0].set_ylabel('Log(PSD)')
axs[0].set_title('Pre-industrial spectra', y=1,fontsize=9)
start_year = 1150
end_year = 2021
for i, column in enumerate(sfrcs.columns):
    sfrcs_data=sfrcs.loc[start_year:end_year][column].values
    x_sfrcs = mtspec.MTSpec(sfrcs_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()

    periods_sfrcs = 1 / x_sfrcs[0]
    axs[1].plot(periods_sfrcs, x_sfrcs[1], label=lab[i],color=colors[i])

axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[0].set_xlim(2,1000)
axs[1].set_xlim(2,1000)

axs[1].grid(True,linestyle='--',color=[0.8,0.8,0.8])
axs[1].legend(frameon=False,fontsize=9,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left')
axs[1].set_xlabel('Log(Period (years))',fontsize=9)
axs[1].set_title('Full period spectra', y=1,fontsize=9)
axs[0].text(1.2,300,'a)')
axs[1].text(1.2,300,'b)')
axs[2].text(1.5,300,'c)')



start_year = 1950
end_year = 2021
for i, column in enumerate(sfrcs.columns):
    sfrcs_data=sfrcs.loc[start_year:end_year][column].values
    x_sfrcs = mtspec.MTSpec(sfrcs_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()

    periods_sfrcs = 1 / x_sfrcs[0]
    axs[2].plot(periods_sfrcs, x_sfrcs[1], label=lab[i],color=colors[i])
og_data=y.loc[start_year:end_year].values
x_og = mtspec.MTSpec(og_data, nw, kspec, dt=1.0, nfft=0, iadapt=0, vn=None, lamb=None).rspec()
periods_og = 1 / x_og[0]
axs[2].plot(periods_og, x_og[1], label='ERA5 temperature',color='k')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlim(2,70)
axs[2].grid(True,linestyle='--',color=[0.8,0.8,0.8])
axs[2].legend(frameon=False,fontsize=9,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left')
axs[2].set_xlabel('Log(Period (years))',fontsize=9)
axs[2].set_title('1950–2021 spectra', y=1,fontsize=9)

plt.tight_layout(w_pad=-0.5)
plt.savefig('spectraMTM.eps',format='eps',bbox_inches='tight')
plt.show()
