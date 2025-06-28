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



recons_pbw = recons_df


f_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'FR_MXD.csv')

og=pd.read_csv(f_path)
og.index=og['age_AD']
ogoverlap=og.loc[1150:2002]['trsgiFir']

ogyhat, ogs3R2c, ogs3RE, ogs3CE = u.long_cps(avg_temp_per_year-273.15, avg_temp_per_year.index, ogoverlap, ogoverlap.index, avg_temp_per_year.loc[1950:2002].index, avg_temp_per_year.loc[1950:2002].index)




def calculate_sea_dataframe(df, events, baseline_years=3, window_years=5):
    # Initialize a dictionary to store mean anomalies for all columns
    mean_anomalies = {column: [] for column in df.columns}
    
    # Define the range of years relative to the event
    relative_years = list(range(-window_years, window_years + 1))
    
    for column in df.columns:
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


events=[1171, 1182, 1191, 1230, 1258, 1276, 1286, 1345, 1453, 1458, 1595, 1601, 1641, 1695, 1783, 1809, 1815, 1832, 1836, 1884, 1991]
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

df = pd.DataFrame(ogyhat)
sea_OG = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
sea_OG['CI_5th'] = ci_5th
sea_OG['CI_95th'] = ci_95th


df = pd.DataFrame(DA2006['RCS recon'])
sea_DA2006 = calculate_sea_dataframe(df, events)
ci_5th, ci_95th = monte_carlo_sea(df, events)
sea_DA2006['CI_5th'] = ci_5th
sea_DA2006['CI_95th'] = ci_95th


sea_summary = calculate_sea_dataframe(recons_pbw, events)
sea_summary




sea_bw = sea_summary[['pbw10', 'pbw20', 'pbw80']]

group1 = ['pbw10', 'pbw20', 'pbw80']
lab=['aMXD 10 $\mu$m','aMXD 20 $\mu$m','aMXD 80 $\mu$m']
width_group1=[1,1,1]
width_group2=[1,1,1]

line_styles_group1 = ['-', '-', '-']
line_styles_group2 = [':', ':', ':']
colors_group1 = ['#D75E6A', '#A21C57', '#49006a']
colors_group2 = ['#D75E6A', '#A21C57', '#49006a']

fig,ax1=plt.subplots(figsize=(3, 2.5))
# Plot group 1
ax1.grid(linestyle='--',alpha=0.5)
ax1.axhline(0, color='black', linestyle='--',linewidth=0.5)
ax1.axvline(0,color='k', linestyle='--',zorder=0)
for i, col in enumerate(group1):
    p=ax1.plot(sea_bw.index, sea_bw[col], label=lab[i], linestyle=line_styles_group1[i],
             color=colors_group1[i],linewidth=width_group1[i],zorder=3)

ax1.plot(sea_10[['CI_5th','CI_95th']],linestyle='--',color='#D75E6A',linewidth=1.5)
ax1.plot(sea_20[['CI_5th','CI_95th']],linestyle='--',color='#A21C57',linewidth=1.5)
ax1.plot(sea_80[['CI_5th','CI_95th']],linestyle='--',color='#49006a',linewidth=1.5)
#plt.plot(sea_bw.index, sea_bw, marker='o')
ax1.axhline(0, color='black', linestyle='--',linewidth=0.5)
ax1.axvline(0,color='k', linestyle='--')
ax1.set_title(f'Volcanic eruption SEA',fontsize=9)
ax1.set_xlabel('Years from peak forcing',fontsize=9)
ax1.set_ylabel('Temperature Anomaly',fontsize=9)
ax1.set_ylim(-2,1)
ax1.set_xlim(-3,5)
ax1.legend(frameon=False, fontsize=8, handlelength=1, borderpad=0,
           ncols=1, columnspacing=0.5, labelspacing=0.2, handletextpad=0.5, loc='lower right')
ax1.set_xticks(ticks=[-3,-2,-1,0,1,2,3,4,5],labels=[-3,-2,-1,0,1,2,3,4,5],fontsize=9)
ax1.set_yticks(ticks=[-1.5,-1,-0.5,0,0.5,1],labels=[-1.5,-1,-0.5,0,0.5,1],fontsize=9)
ax1.text(5.1,0.8,'95%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')
ax1.text(5.1,-0.85,'5%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')

plt.savefig('SEA_v2.eps', format='eps',bbox_inches='tight')




################################################
# supp figure

sea_bw = sea_summary[['pbw10', 'pbw20', 'pbw80','bw10', 'bw20', 'bw80']]

group1 = ['pbw10', 'pbw20', 'pbw80']
group2 = ['bw10', 'bw20', 'bw80']
lab=['aMXD10','aMXD20','aMXD80']
width_group1=[1,1,1]
width_group2=[1,1,1]

line_styles_group1 = ['-', '-', '-']
line_styles_group2 = [':', ':', ':']
colors_group1 = ['#D75E6A', '#A21C57', '#49006a']
colors_group2 = ['#D75E6A', '#A21C57', '#49006a']

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(5.5, 2.5),sharey=True)
# Plot group 1

for i, col in enumerate(group1):
    p=ax1.plot(sea_bw.index, sea_bw[col], label=lab[i], linestyle=line_styles_group1[i],
             color=colors_group1[i],linewidth=width_group1[i])
# Plot group 2
for i, col in enumerate(group2):
   nop=ax1.plot(sea_bw.index, sea_bw[col],label=f'{col}', linestyle=line_styles_group2[i], 
             color=colors_group2[i],linewidth=width_group2[i])
ax1.plot(sea_10[['CI_5th','CI_95th']],linestyle='--',color='#D75E6A',linewidth=1)
ax1.plot(sea_20[['CI_5th','CI_95th']],linestyle='--',color='#A21C57',linewidth=1)
ax1.plot(sea_80[['CI_5th','CI_95th']],linestyle='--',color='#49006a',linewidth=1)
#plt.plot(sea_bw.index, sea_bw, marker='o')
ax1.axhline(0, color='black', linestyle='--',linewidth=0.5)
ax1.axvline(0,color='k', linestyle='--')
ax1.set_title(f'Biweight aggregation',fontsize=9)
ax1.set_xlabel('Years from peak forcing',fontsize=9)
ax1.set_ylabel('Temperature Anomaly',fontsize=9)
ax1.set_ylim(-2,1)
ax1.set_xlim(-3,5)
ax1.legend(frameon=False, fontsize=8, handlelength=1, borderpad=0,
           ncols=2, columnspacing=0.5, labelspacing=0.5, handletextpad=0.5, loc='lower right')
ax1.set_xticks(ticks=[-3,-2,-1,0,1,2,3,4,5],labels=[-3,-2,-1,0,1,2,3,4,5],fontsize=9)
ax1.set_yticks(ticks=[-1.5,-1,-0.5,0,0.5,1],labels=[-1.5,-1,-0.5,0,0.5,1],fontsize=9)
ax1.grid(linestyle='--',alpha=0.5)
ax1.text(-3,1.1,'a)',fontsize=9)
ax1.text(5.1,0.8,'95%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')
ax1.text(5.1,-0.85,'5%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')



sea_bw = sea_summary[['pq10', 'pq20', 'pq80','q10','q20','q80']]

group1 = ['pq10', 'pq20', 'pq80']
group2 = ['q10','q20','q80']


for i, col in enumerate(group1):
    p=ax2.plot(sea_bw.index, sea_bw[col], label=f'{col}', linestyle=line_styles_group1[i],
             color=colors_group1[i],linewidth=width_group1[i])
# Plot group 2
for i, col in enumerate(group2):
    nop=ax2.plot(sea_bw.index, sea_bw[col], label=f'{col}', linestyle=line_styles_group2[i], 
             color=colors_group2[i],linewidth=width_group2[i])
#plt.plot(sea_bw.index, sea_bw, marker='o')
ax2.plot(seaq_10[['CI_5th','CI_95th']],linestyle='--',color='#D75E6A',linewidth=1)
ax2.plot(seaq_20[['CI_5th','CI_95th']],linestyle='--',color='#A21C57',linewidth=1)
ax2.plot(seaq_80[['CI_5th','CI_95th']],linestyle='--',color='#49006a',linewidth=1)
ax2.axhline(0, color='black', linestyle='--',linewidth=0.5)
ax2.axvline(0,color='k', linestyle='--')
ax2.set_title(f'Q75 aggregation',fontsize=9)
ax2.set_xlabel('Years from peak forcing',fontsize=9)
#ax2.set_ylabel('Temperature anomaly from background mean',fontsize=9)
ax2.set_ylim(-2,1)
ax2.set_xlim(-3,5)
ax2.legend(frameon=False, fontsize=8, handlelength=1, borderpad=0,
           ncols=2, columnspacing=0.5, labelspacing=0.5, handletextpad=0.5, loc='lower right')
ax2.set_xticks(ticks=[-3,-2,-1,0,1,2,3,4,5],labels=[-3,-2,-1,0,1,2,3,4,5],fontsize=9)
ax2.set_yticks(ticks=[-1.5,-1,-0.5,0,0.5,1],labels=[-1.5,-1,-0.5,0,0.5,1],fontsize=9)
ax2.grid(linestyle='--',alpha=0.5)
ax2.text(-3,1.1,'b)',fontsize=9)
ax2.text(5.1,0.8,'95%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')
ax2.text(5.1,-0.85,'5%',fontsize=8,horizontalalignment='left',
        verticalalignment='center')
plt.tight_layout(w_pad=-.4)
plt.savefig('SEAfull.eps',format='eps',bbox_inches='tight')
plt.show()
