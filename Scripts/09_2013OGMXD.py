import pandas as pd
from csaps import csaps
import numpy as np
from math import cos, pi
from sklearn.preprocessing import StandardScaler
import os
import netCDF4 as nc
from datetime import datetime, timedelta
import utils as u  
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
import pymannkendall as mk
from sklearn.preprocessing import scale
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import acf


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
sfrcs_filtered = sfrcs_p.loc[1950:2021]

pbwfilt=pd.DataFrame(None)
order = 2  # Filter order
cutoff_period = 10  # desired cutoff period in years
cutoff = 1 / cutoff_period  
nyquist = 0.5 
normal_cutoff = cutoff / nyquist

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
        normalR[column]=rn
    return lowR.iloc[0],normalR.iloc[0]

H10,nr=filtcorr(1/10,'high')
H30,nr=filtcorr(1/30,'high')

L5,nr=filtcorr(1/5,'low')
L10,normalR=filtcorr(1/10,'low')
L20,normalR=filtcorr(1/20,'low')


r10=[H10['pbw10'],H30['pbw10'],normalR['pbw10'],L5['pbw10'],L20['pbw10']]
r20=[H10['pbw20'],H30['pbw20'],normalR['pbw20'],L5['pbw20'],L20['pbw20']]
r40=[H10['pbw40'],H30['pbw40'],normalR['pbw40'], L5['pbw40'],L20['pbw40']]
r80=[H10['pbw80'],H30['pbw80'],normalR['pbw80'],L5['pbw80'],L20['pbw80']]

def adjusted_significance(series_x, series_y, critical_value=2.58):
    N = len(series_x)
    AC_x = acf(series_x, nlags=1)[1]
    AC_y = acf(series_y, nlags=1)[1]

    if AC_x >= 0.98 or AC_y >= 0.98: 
        return 1.0  
    Ne = round(N * ((1 - AC_x * AC_y) / (1 + AC_x * AC_y)))
    sig_threshold = critical_value / np.sqrt(Ne - 2) #crit value for p<0.01
    return sig_threshold

cutoff= 1/7
b, a = butter(order, cutoff, btype='low', analog=False,fs=1)
Augfilt = filtfilt(b, a, y)
for column in sfrcs_filtered.columns:
    pbwfilt[column]=filtfilt(b, a, sfrcs_filtered[column])
series_x = pbwfilt['pbw10'] 
series_y = Augfilt  

sig_level = adjusted_significance(series_x, series_y)

print(sig_level)



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


f_path = '/Users/julieedwards/Documents/Projects/MANCHA/MXD/FR_MXD.csv'

og=pd.read_csv(f_path)
og.index=og['age_AD']
ogoverlap=og.loc[1150:2002]['trsgiFir']

ogyhat, ogs3R2c, ogs3RE, ogs3CE = u.long_cps(avg_temp_per_year-273.15, avg_temp_per_year.index, ogoverlap, ogoverlap.index, avg_temp_per_year.loc[1950:2002].index, avg_temp_per_year.loc[1950:2002].index)

subrecons_df=recons_df[['pq10','pq20','pq40','pq80','pq100']]
subreconsbw_df=recons_df[['pbw10','pbw20','pbw40','pbw80','pbw100']]

ogr=[]
ogbwr=[]
for column in subrecons_df.columns:
    r=pearsonr(ogoverlap,subrecons_df.loc[1150:2002][column])
    ogr.append(r.statistic)
for column in subreconsbw_df.columns:
    rbw=pearsonr(ogoverlap,subreconsbw_df.loc[1150:2002][column])
    ogbwr.append(rbw.statistic)
fig=plt.subplots(figsize=(3.5,3))
plt.plot(ogbwr,'-o',color='k')
plt.plot(ogr,':o',color='k')
plt.ylim(0.75,0.9)
plt.xticks([0,1,2,3,4],labels=['aMXD10','aMXD20','aMXD40','aMXD80','aMXD100'],fontsize=9)
plt.grid()
plt.title('Correlation with 2013MXD')
plt.ylabel('R')
plt.savefig('2013corrp.eps',format='eps',bbox_inches='tight')


result = mk.original_test(avg_temp_per_year)
temp_slp = result.slope

rmse = results_df.loc[results_df['Proxy'] == 'pbw10', 'RMSE'].values[0]  # Assuming there's only one RMSE value per Proxy
num_simulations = 1000
simulations = np.zeros((num_simulations, len(recons_df.loc[1150:2021]['pbw10'])))

# Perform Monte Carlo simulations
for i in range(num_simulations):
    noise = np.random.normal(0, rmse, len(recons_df.loc[1150:2021]['pbw10']))
    simulations[i] = recons_df.loc[1150:2021]['pbw10'] + noise

# Calculate the 95% confidence interval bounds
lower_bound = np.percentile(simulations, 2.5, axis=0)
upper_bound = np.percentile(simulations, 97.5, axis=0)

import matplotlib.gridspec as gridspec
fig=plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1,1], width_ratios=[1, 1.2, 1])
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(avg_temp_per_year-273.15,color='k',label='ERA5',linewidth=2,linestyle='-')
ax1.plot(recons_df.loc[1900:2021]['pbw10'],color='#D75E9B',label='aMXD 10 $\mu$m',linewidth=1)
ax1.plot(recons_df.loc[1900:2021]['pbw20'],color='#A21C57',label='aMXD 20 $\mu$m',linewidth=1)
ax1.plot(recons_df.loc[1900:2021]['pbw80'],color='#660C23',label='aMXD 80 $\mu$m',linewidth=1)
ax1.set_ylim(4,20)
ax1.set_xlim(1950,2021)
ax1.set_xticks([1950,1960,1970,1980,1990,2000,2010,2020])
ax1.set_xticklabels([1950,1960,1970,1980,1990,2000,2010,2020],fontsize=9)
ax1.set_ylabel('Temperature ($^\circ$C)',fontsize=9)
ax1.grid(linestyle='--',alpha=0.5)
ax1.legend(frameon=False,fontsize=8,handlelength=1, columnspacing=0.5,labelspacing=0.5,
           handletextpad=0.5,ncols=2,loc='upper left') 
ax1.text(1998,18,'10 $\mu$m R$^2$=0.64',color='#D75E9B',fontsize=9)
ax1.text(1998,16,'20 $\mu$m R$^2$=0.61',color='#A21C57',fontsize=9)
ax1.text(1998,14,'80 $\mu$m R$^2$=0.48',color='#660C23',fontsize=9)
ax1.text(1940,20,'a)',fontsize=9)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(r10,'-o',label='aMXD10',color='#D75E9B')
ax3.plot(r20,'-o',label='aMXD20',color='#A21C57')
ax3.plot(r80,'-o',label='aMXD80',color='#660C23')
ax3.set_yticks([0.6,0.7,0.8],labels=[0.6,0.7,0.8],fontsize=9)
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax3.text(-0.8,0.83,'b)',fontsize=9)
ax3.set_ylabel('R', fontsize=9, rotation=0, labelpad=10, ha='center')
#ax3.set_xlabel('High-pass filter (n years)  Low-pass filter (n years)',fontsize=9)
ax3.grid(linestyle='--',alpha=0.5)
ax3.set_xticks([0,1,2,3,4],labels=['H10','H30',' None','L5','L20'],fontsize=9)
plt.legend(frameon=False,fontsize=8,handlelength=1,borderpad=0,
           labelspacing=0.3,handletextpad=0.5,loc='lower left')
splineamount=100
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw10'],splineamount),label='aMXD 10 $\mu$m',color='#D75E9B')
ax2.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw20'],splineamount),label='aMXD 20 $\mu$m',color='#A21C57')
ax2.plot(recons_df.loc[1150:2021].index,spline(recons_df.loc[1150:2021].index,recons_df.loc[1150:2021]['pbw80'],splineamount),label='aMXD 80 $\mu$m',color='#660C23')
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
ax2.grid(linestyle='--',alpha=0.5)
ax2.set_title('Reconstructed mid-July to mid-August temperature (100-year spline)',fontsize=9)
ax4 = fig.add_subplot(gs[2, :])
pbw10_series = recons_df.loc[1150:2021, 'pbw10']
rmse_value = results_df.loc[results_df['Proxy'] == 'pbw10', 'RMSE'].values[0]  # Assuming there's only one RMSE value per Proxy
unc=ax4.fill_between(pbw10_series.index,
                     pbw10_series - rmse_value,
                     pbw10_series + rmse_value,
                     color='#D75E9B',label='$\pm$ RMSE',alpha=0.5)
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
ax4.grid(linestyle='--',alpha=0.5)
ax4.set_title('Reconstructed mid-July to mid-August temperature',fontsize=9)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.5)
plt.savefig('Reconstructionpanels_filter20.eps', format='eps',bbox_inches='tight')
plt.show()


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
plt.savefig('2013MXDaMXD.eps', format='eps',bbox_inches='tight')
##Forcing and response are calculated relative 
# to a pre-event 5-year background period undisturbed 
# by volcanic forcing (e.g., 1804–1808 for the 1809 and 1815 volcanic eruptions, respectively)


#eVolv2k_v3 VSSI>6
import pandas as pd
import matplotlib.pyplot as plt

# Sample data structure
# Assuming recons_df is already defined
recons_pbw = recons_df
file='/Users/julieedwards/Documents/Projects/MANCHA/Climate/eVolv2k_v3_ds_1.nc'
dataset = nc.Dataset(file)
vssi = dataset.variables['vssi'][:]
time = dataset.variables['yearCE'][:]


#Sigl events >2 forcing
events=[1258, 1458, 1815, 1230, 1783, 1809, 1641, 1601, 1171, 1695, 1286, 1345, 1276, 1836, 1991, 1832, 1453, 1191, 1595, 1182, 1884, 1585, 1210, 1862, 1964, 1512, 1762, 1269, 1912, 1729, 1673, 1477, 1329, 1389, 1739, 1667]
#>3 forcing
events=[1258, 1458, 1815, 1230, 1783, 1809, 1641, 1601, 1171, 1695, 1286, 1345, 1276, 1836, 1991, 1832, 1453, 1191, 1595, 1182, 1884, 1585, 1210, 1862, 1964, 1512, 1762, 1269, 1912, 1729, 1673, 1477]
#>=Krakatoa
events=[1258, 1458, 1815, 1230, 1783, 1809, 1641, 1601, 1171, 1695, 1286, 1345, 1276, 1836, 1991, 1832, 1453, 1191, 1595, 1182, 1884]
#events=np.array(time[vssi>6])
#events=np.flip(events[events>1150])

#Toohey2017 VSSI greater than 2 TgS. 
# exclude events for which an eruption of magnitude greater than 2 TgS occurred within the preceding six years, 
# and also those for which an event greater than 10 TgS occurred within the preceding ten years. 
# effusive eruption style, are also excluded 
#events=[1170,1229,1257,1275,1343,1388,1452,1553,1586,1594,
#        1640,1653,1762,1809,1831,1861,1883,
#        1902,1963,1982,1991]
import pandas as pd
import numpy as np

def calculate_sea_dataframe(df, events, baseline_years=3, window_years=5):
    # Initialize a dictionary to store mean anomalies for all columns
    mean_anomalies = {column: [] for column in df.columns}
    
    # Define the range of years relative to the event
    relative_years = list(range(-window_years, window_years + 1))
    
    for column in df.columns:
        # Initialize a list to store anomalies for each relative year
        anomalies = {year: [] for year in relative_years}
        
        for event in events:
            # Calculate the baseline mean for the event (e.g., 3 years before the event)
            start_baseline = event - baseline_years
            end_baseline = event - 1
            baseline_values = df[(df.index >= start_baseline) & (df.index <= end_baseline)][column]
            baseline_mean = baseline_values.mean() if not baseline_values.empty else np.nan
            
            for i in relative_years:
                year = event + i
                if year in df.index:
                    value = df.loc[year, column]
                    anomaly = value - baseline_mean
                else:
                    anomaly = np.nan
                
                anomalies[i].append(anomaly)
        
        # Calculate the mean anomaly for each relative year and store it in the mean_anomalies dictionary
        for year in relative_years:
            mean_anomalies[column].append(np.nanmean(anomalies[year]))
    
    # Create a DataFrame from the mean anomalies dictionary
    sea_summary = pd.DataFrame(mean_anomalies, index=relative_years)
    
    return sea_summary

def calculate_sea_matrix(df, events, baseline_years=3, window_years=5):
    relative_years = list(range(-window_years, window_years + 1))
    sea_matrix = np.full((len(events), len(relative_years)), np.nan)
    
    for idx, event in enumerate(events):
        baseline_start = event - baseline_years
        baseline_end = event - 1
        baseline_values = df[(df.index >= baseline_start) & (df.index <= baseline_end)].mean()
        for i, year in enumerate(relative_years):
            target_year = event + year
            if target_year in df.index:
                sea_matrix[idx, i] = df.loc[target_year].mean() - baseline_values.mean()
    
    return sea_matrix

def monte_carlo_sea(df, events, baseline_years=3, window_years=5, num_simulations=1000):
    relative_years = list(range(-window_years, window_years + 1))
    n_events = len(events)
    
    sea_mbar = np.full((num_simulations, len(relative_years)), np.nan)
    
    for i in range(num_simulations):
        random_events = np.random.choice(df.index, n_events, replace=False)
        sea_matrix = calculate_sea_matrix(df, random_events, baseline_years, window_years)
        sea_mbar[i, :] = np.nanmean(sea_matrix, axis=0)
    
    tci = np.sort(sea_mbar, axis=0)
    ci_9th = tci[int(np.floor(num_simulations * 0.05)), :]
    ci_95th = tci[int(np.floor(num_simulations * 0.95)), :]
    
    return ci_9th, ci_95th

# Example usage

events=[1258, 1458, 1815, 1230, 1783, 1809, 1641, 1601, 1171, 1695, 1286, 1345, 1276, 1836, 1991, 1832, 1453, 1191, 1595, 1182, 1884]

df = pd.DataFrame(recons_df['pbw10'])
sea_10 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
sea_10['CI_5th'] = ci_5th
sea_10['CI_95th'] = ci_95th

df = pd.DataFrame(recons_df['pbw20'])
sea_20 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
sea_20['CI_5th'] = ci_5th
sea_20['CI_95th'] = ci_95th

df = pd.DataFrame(recons_df['pbw80'])
sea_80 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
sea_80['CI_5th'] = ci_5th
sea_80['CI_95th'] = ci_95th

df = pd.DataFrame(recons_df['pq10'])
seaq_10 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
seaq_10['CI_5th'] = ci_5th
seaq_10['CI_95th'] = ci_95th

df = pd.DataFrame(recons_df['pq20'])
seaq_20 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
seaq_20['CI_5th'] = ci_5th
seaq_20['CI_95th'] = ci_95th

df = pd.DataFrame(recons_df['pq80'])
seaq_80 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
seaq_80['CI_5th'] = ci_5th
seaq_80['CI_95th'] = ci_95th

og.index=og['age_AD']
ogJA=pd.DataFrame(og['trsgiFir'])

seaog=calculate_sea_dataframe(ogJA, events)

# Example usage
# Assuming recons_pbw is your DataFrame and events is your list of events
# recons_pbw = ...
# events = ...
sea_summary = calculate_sea_dataframe(recons_pbw, events)
sea_summary






sea_bw = sea_summary[['pbw10', 'pbw20', 'pbw80']]

group1 = ['pbw10', 'pbw20', 'pbw80']
lab=['aMXD 10 $\mu$m','aMXD 20 $\mu$m','aMXD 80 $\mu$m']
width_group1=[1,1,1]
width_group2=[1,1,1]

line_styles_group1 = ['-', '-', '-']
line_styles_group2 = [':', ':', ':']
colors_group1 = ['#D75E9B', '#A21C57', '#660C23']
colors_group2 = ['#D75E9B', '#A21C57', '#660C23']

fig,ax1=plt.subplots(1,1,figsize=(3, 2.5),sharey=True)
# Plot group 1
ax1.grid(linestyle='--',alpha=0.5)
ax1.axhline(0, color='black', linestyle='--',linewidth=0.5)
ax1.axvline(0,color='k', linestyle='--',zorder=0)
for i, col in enumerate(group1):
    p=ax1.plot(sea_bw.index, sea_bw[col], label=lab[i], linestyle=line_styles_group1[i],
             color=colors_group1[i],linewidth=width_group1[i],zorder=3)
# Plot group 2
#for i, col in enumerate(group2):
 #   nop=ax1.plot(sea_bw.index, sea_bw[col],label=f'{col}', linestyle=line_styles_group2[i], 
 #            color=colors_group2[i],linewidth=width_group2[i])
 
ax1.plot(sea_10[['CI_5th','CI_95th']],linestyle='--',color='#D75E9B',linewidth=1.5)
ax1.plot(sea_20[['CI_5th','CI_95th']],linestyle='--',color='#A21C57',linewidth=1.5)
ax1.plot(sea_80[['CI_5th','CI_95th']],linestyle='--',color='#660C23',linewidth=1.5)
#plt.plot(sea_bw.index, sea_bw, marker='o')
ax1.axhline(0, color='black', linestyle='--',linewidth=0.5)
ax1.axvline(0,color='k', linestyle='--')
ax1.set_title(f'Volcanic eruption SEA',fontsize=9)
ax1.set_xlabel('Years from peak forcing',fontsize=9)
ax1.set_ylabel('Temperature Anomaly',fontsize=9)
ax1.set_ylim(-2,1)
ax1.set_xlim(-3,5)
ax1.legend(frameon=False, fontsize=8, handlelength=1, borderpad=0,
           ncols=1, columnspacing=0.5, labelspacing=0.5, handletextpad=0.5, loc='lower right')
ax1.set_xticks(ticks=[-3,-2,-1,0,1,2,3,4,5],labels=[-3,-2,-1,0,1,2,3,4,5],fontsize=9)
ax1.set_yticks(ticks=[-1.5,-1,-0.5,0,0.5,1],labels=[-1.5,-1,-0.5,0,0.5,1],fontsize=9)
ax1.text(5.1,0.8,'95%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')
ax1.text(5.1,-0.85,'5%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')
plt.savefig('SEA_v2.eps', format='eps',bbox_inches='tight')

