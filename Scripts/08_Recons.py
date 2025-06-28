import pandas as pd
from csaps import csaps
import numpy as np
from math import cos, pi
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
from scipy.signal import coherence
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


runnin20=recons_df['pbw10'].rolling(window=20).mean()
fig, ax = plt.subplots(figsize=(4, 3))  
plt.grid(linestyle='--',zorder=1)
sns.histplot(runnin20, kde=False, stat="density", color='grey', alpha=0.6, ax=ax, binwidth=0.25,zorder=2)
sns.kdeplot(runnin20, cumulative=False, color='k', ax=ax)
sns.kdeplot(runnin20, cumulative=True, color='red', ax=ax,clip=(runnin20.min(),runnin20.max()))
ax.set_xlabel('Reconstructed Temperature ($^\circ$C)')
ax.set_ylabel('Density')
ax.set_title('2002-2021 is hottest 20-year period',fontsize=9)
ax.text(10.35, 0.6, '2002-2021',fontsize=9,rotation=90)
ax.axvline(runnin20.loc[2021], linestyle='--', color='k')
plt.legend(frameon=False,fontsize=9)
plt.xticks([6,7,8,9,10,11])
plt.xlim(5,11)
plt.tight_layout()
plt.show()







# subset to the time coverage of the ERA5 data (1950 start year)
sfrcs_filtered = sfrcs_p.loc[1950:2021]
# filter correlations
pbwfilt=pd.DataFrame(None)
order = 2  # Filter order
cutoff_period = 10  # desired cutoff period in years
cutoff = 1 / cutoff_period  # default


def filtcorr(cutoff,type):
    b, a = butter(order, cutoff, btype=type, analog=False,fs=1)
    Augfilt = filtfilt(b, a, y)
    for column in sfrcs_filtered.columns:
        pbwfilt[column]=filtfilt(b, a, sfrcs_filtered[column])
    lowR=pd.DataFrame(None)
    normalR=pd.DataFrame(None)
    for column in sfrcs_filtered.columns:
        r=pearsonr(Augfilt,pbwfilt[column])
        lowR[column]=r
        rn=pearsonr(y,sfrcs_filtered[column])
        normalR[column]=rn # calculates correlation without filter on either time series
    return lowR.iloc[0],normalR.iloc[0]

H10,_=filtcorr(1/10,'high')
H30,_=filtcorr(1/30,'high')
L5,_=filtcorr(1/5,'low')
L20,normalR=filtcorr(1/20,'low')

#setting up data to fit plotting code
r10=[H10['pbw10'],H30['pbw10'],normalR['pbw10'],L5['pbw10'],L20['pbw10']]
r20=[H10['pbw20'],H30['pbw20'],normalR['pbw20'],L5['pbw20'],L20['pbw20']]
r40=[H10['pbw40'],H30['pbw40'],normalR['pbw40'], L5['pbw40'],L20['pbw40']]
r80=[H10['pbw80'],H30['pbw80'],normalR['pbw80'],L5['pbw80'],L20['pbw80']]


sig_level=pd.DataFrame()
# To determine when (at what lowpass cutoff period) the autocorrelation of the filter time series prevents significance
def adjusted_significance(series_x, series_y, critical_value=2.58):
    N = len(series_x)
    AC_x = acf(series_x, nlags=1)[1]
    AC_y = acf(series_y, nlags=1)[1]

    if AC_x >= 0.98 or AC_y >= 0.98: 
        return 1.0  
    Ne = round(N * ((1 - AC_x * AC_y) / (1 + AC_x * AC_y)))
    sig_threshold = critical_value / np.sqrt(Ne - 2) #crit value for p<0.01
    return sig_threshold

cutoff= 1/5
b, a = butter(order, cutoff, btype='low', analog=False,fs=1)
Augfilt = filtfilt(b, a, y)
for column in sfrcs_filtered.columns:
    pbwfilt[column]=filtfilt(b, a, sfrcs_filtered[column])
series_x = pbwfilt['pbw10'] 
series_y = Augfilt  
adjsig = adjusted_significance(series_x, series_y)
sig_level.loc[0,'5']=adjsig

cutoff= 1/6
b, a = butter(order, cutoff, btype='low', analog=False,fs=1)
Augfilt = filtfilt(b, a, y)
for column in sfrcs_filtered.columns:
    pbwfilt[column]=filtfilt(b, a, sfrcs_filtered[column])
series_x = pbwfilt['pbw10'] 
series_y = Augfilt  
adjsig = adjusted_significance(series_x, series_y)
sig_level.loc[0,'6']=adjsig

cutoff= 1/20
b, a = butter(order, cutoff, btype='low', analog=False,fs=1)
Augfilt = filtfilt(b, a, y)
for column in sfrcs_filtered.columns:
    pbwfilt[column]=filtfilt(b, a, sfrcs_filtered[column])
series_x = pbwfilt['pbw10'] 
series_y = Augfilt  
adjsig = adjusted_significance(series_x, series_y)
sig_level.loc[0,'20']=adjsig



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




## plotting
fig=plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1,1], width_ratios=[1, 1, 1])
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(avg_temp_per_year-273.15,color='k',label='ERA5',linewidth=2,linestyle='-')
ax1.plot(recons_df.loc[1900:2021]['pbw10'],color='#D75E6A',label='aMXD 10 $\mu$m',linewidth=1,zorder=5)
ax1.plot(recons_df.loc[1900:2021]['pbw20'],color='#A21C57',label='aMXD 20 $\mu$m',linewidth=1)
ax1.plot(recons_df.loc[1900:2021]['pbw80'],color='#49006a',label='aMXD 80 $\mu$m',linewidth=1)
ax1.set_ylim(4,20)
ax1.set_xlim(1950,2021)
ax1.set_xticks([1950,1960,1970,1980,1990,2000,2010,2020])
ax1.set_xticklabels([1950,1960,1970,1980,1990,2000,2010,2020],fontsize=9)
ax1.set_ylabel('Temperature ($^\circ$C)',fontsize=9)
ax1.grid(linestyle='--',color=[0.8,0.8,0.8])
ax1.legend(frameon=False,fontsize=8,handlelength=1, columnspacing=0.5,labelspacing=0.5,
           handletextpad=0.5,ncols=2,loc='upper left') 
ax1.text(1997.5,18,'10 $\mu$m R$^2$=0.64',color='#D75E6A',fontsize=9)
ax1.text(1997.5,16,'20 $\mu$m R$^2$=0.61',color='#A21C57',fontsize=9)
ax1.text(1997.5,14,'80 $\mu$m R$^2$=0.48',color='#49006a',fontsize=9)
ax1.text(1939,20,'a)',fontsize=9)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(r10,'-o',label='aMXD10',color="#D75E6A")
ax3.plot(r20,'-o',label='aMXD20',color='#A21C57')
ax3.plot(r80,'-o',label='aMXD80',color='#49006a')
ax3.axvline(3.5)
ax3.set_yticks([0.6,0.7,0.8],labels=[0.6,0.7,0.8],fontsize=9)
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax3.text(-0.8,0.83,'b)',fontsize=9)
ax3.set_ylabel('R', fontsize=9, rotation=0, labelpad=10, ha='center')
#ax3.set_xlabel('High-pass filter (n years)  Low-pass filter (n years)',fontsize=9)
ax3.grid(linestyle='--',color=[0.8,0.8,0.8])
ax3.set_xticks([0,1,2,3,4],labels=['H10','H30',' None','L5','L20'],fontsize=9)
plt.legend(frameon=False,fontsize=8,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='lower left')
splineamount=100
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw10'],splineamount),label='aMXD 10 $\mu$m',color="#D75E6A")
ax2.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw20'],splineamount),label='aMXD 20 $\mu$m',color='#A21C57')
ax2.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw80'],splineamount),label='aMXD 80 $\mu$m',color='#49006a')
ax2.axvspan(1950,2021,color='gray',alpha=0.3)
ax2.text(1070,12.5,'c)',fontsize=9)
ax2.set_ylim(6,12)
ax2.set_xlim(1150,2021)
ax2.legend(frameon=False,fontsize=8,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left') # Adjust the size as needed
ax2.set_xticks([1200,1300,1400,1500,1600,1700,1800,1900,2000])
ax2.set_xticklabels([1200,1300,1400,1500,1600,1700,1800,1900,2000],fontsize=9)
ax2.set_yticklabels([6,8,10,12],fontsize=9)
ax2.set_xlabel('Year',fontsize=9)
ax2.set_ylabel('Temperature ($^\circ$C)',fontsize=9)
ax2.grid(linestyle='--',color=[0.8,0.8,0.8])
ax2.set_title('Reconstructed mid-July to mid-August temperature (100-year spline)',fontsize=9)
ax4 = fig.add_subplot(gs[2, :])
pbw10_series = recons_df.loc[1150:2021, 'pbw10']
rmse_value = results_df.loc[results_df['Proxy'] == 'pbw10', 'RMSE'].values[0]  # Assuming there's only one RMSE value per Proxy

unc=ax4.fill_between(pbw10_series.index,
                     pbw10_series - rmse_value,
                     pbw10_series + rmse_value,
                     color='#D75E6A',label='$\pm$ RMSE',alpha=0.5)
ax4.plot(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw10'],label='aMXD 10 $\mu$m',color='k',linewidth=0.5)
ax4.set_xlim(1150,2021)
ax4.legend(frameon=False,fontsize=8,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='upper left') # Adjust the size as needed
ax4.set_xticks([1200,1300,1400,1500,1600,1700,1800,1900,2000])
ax4.set_xticklabels([1200,1300,1400,1500,1600,1700,1800,1900,2000],fontsize=9)
ax4.set_ylim(-2,17)
ax4.set_yticks([0,5,10,15])
ax4.text(1070,18,'d)',fontsize=9)
ax4.set_yticklabels([0,5,10,15],fontsize=9)
ax4.set_xlabel('Year',fontsize=9)
ax4.set_ylabel('Temperature ($^\circ$C)',fontsize=9)
ax4.grid(linestyle='--',color=[0.8,0.8,0.8])
ax4.set_title('Reconstructed mid-July to mid-August temperature',fontsize=9)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', 'ReconsFigure.eps'), format='eps',bbox_inches='tight')
plt.show()



