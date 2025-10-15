import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import netCDF4 as nc
from datetime import datetime, timedelta
import utils as u  
import matplotlib.pyplot as plt
import pymannkendall as mk
from sklearn.preprocessing import scale
from scipy.stats import spearmanr
import xarray as xr
import seaborn as sns


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
    #s3R2v  = rhov[0,1] ** 2 # square the off-diagonal element
    s3R2c  = rhoc[0,1] ** 2 # square the off-diagonal element
    return yhat, s3R2c,s3RE, s3CE

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

datasets=[]
directory_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'raw')
for filename in os.listdir(directory_path):
    if filename.startswith('col'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, index_col='Unnamed: 0')
        df.index.name = 'YEAR'
        res=os.path.basename(file_path)
        foo=df.to_xarray()  
        foo = foo.assign_coords(res=res)
        
        datasets.append(foo)

combined_dataarray = xr.concat(datasets, dim='res')

aMXD10 = combined_dataarray.sel(res='colpbw10.ind').to_dataframe().drop(columns='res')
aMXD20 = combined_dataarray.sel(res='colpbw20.ind').to_dataframe().drop(columns='res')
aMXD40 = combined_dataarray.sel(res='colpbw40.ind').to_dataframe().drop(columns='res')
aMXD80 = combined_dataarray.sel(res='colpbw80.ind').to_dataframe().drop(columns='res')
aMXD100 = combined_dataarray.sel(res='colpbw100.ind').to_dataframe().drop(columns='res')



rw=pd.read_csv('/Users/julieedwards/Documents/Projects/MANCHA/toGit/Firth/Data/QWA/raw/MRW_biweight_10mu.txt',delim_whitespace=True)
rw.index=rw['YEAR']
rw=rw.drop(columns=['YEAR'])
rw.columns = [col.replace('_', '') for col in rw.columns]
rw.columns = [col[0:6] for col in rw.columns]
rw_aligned = rw.reindex(columns=aMXD10.columns)


def compute_spearman(df1, df2, start_year):
    corre_dict = {}
    # Loop :calculate the Spearman correlation
    for column in df1.columns:
        corre_dict[column] = spearmanr(
            df1.loc[start_year:][column],  
            df2.loc[start_year:][column],  
            nan_policy='omit'  
        ).statistic
    corre_df = pd.DataFrame(corre_dict, index=[0])
    return corre_df



corre10 = compute_spearman(aMXD10, rw_aligned, 1150)
corre20 = compute_spearman(aMXD20, rw_aligned, 1150)
corre40 = compute_spearman(aMXD40, rw_aligned, 1150)
corre80 = compute_spearman(aMXD80, rw_aligned, 1150)
corre100 = compute_spearman(aMXD100, rw_aligned, 1150)

all_corre = pd.concat([corre10, corre20, corre40, corre80, corre100], axis=0, ignore_index=False)

all_corre.index = ['corre10', 'corre20', 'corre40', 'corre80', 'corre100']

corre_all=all_corre.T

sp_results = {}
for column in corre_all:
    sp_results[column] = corre_all[column].mean()
sp_df = pd.DataFrame(list(sp_results.items()), columns=['Column', 'Spearman_Coefficient'])




rwindex=rw_aligned<200

indus=pd.DataFrame()
indus['aMXD10']=aMXD10.mean(axis=1)
indus['aMXD20']=aMXD20.mean(axis=1)
indus['aMXD40']=aMXD40.mean(axis=1)
indus['aMXD80']=aMXD80.mean(axis=1)
indus['aMXD100']=aMXD100.mean(axis=1)
Recons = {}
for proxy_column in indus:
    proxy = indus[[proxy_column]]
    scaler = StandardScaler()
    proxy_scaled = scaler.fit_transform(proxy)
    yhat, s3R2c, s3RE, s3CE = long_cps(y, y.index, proxy_scaled, indus.index, y.index, y.index)
    Recons[proxy_column] = yhat
recons_df = pd.DataFrame(Recons)

mk_results = {}
for column in recons_df.columns:
    result = mk.original_test(recons_df.loc[1800:][column])
    mk_results[column] = result.slope
mk_dfall = pd.DataFrame(list(mk_results.items()), columns=['Column', 'Mann-Kendall_Slope_Coefficient'])

indus=pd.DataFrame()
indus['aMXD10']=aMXD10[rwindex].mean(axis=1)
indus['aMXD20']=aMXD20[rwindex].mean(axis=1)
indus['aMXD40']=aMXD40[rwindex].mean(axis=1)
indus['aMXD80']=aMXD80[rwindex].mean(axis=1)
indus['aMXD100']=aMXD100[rwindex].mean(axis=1)
Recons = {}
for proxy_column in indus:
    proxy = indus[[proxy_column]]
    scaler = StandardScaler()
    proxy_scaled = scaler.fit_transform(proxy)
    yhat, s3R2c, s3RE, s3CE = long_cps(y, y.index, proxy_scaled, indus.index, y.index, y.index)
    Recons[proxy_column] = yhat
recons_df = pd.DataFrame(Recons)




mk_results = {}
for column in recons_df.columns:
    result = mk.original_test(recons_df.loc[1800:][column])
    mk_results[column] = result.slope
mk_df300 = pd.DataFrame(list(mk_results.items()), columns=['Column', 'Mann-Kendall_Slope_Coefficient'])



norm = plt.Normalize(mk_dfall['Mann-Kendall_Slope_Coefficient'].min(), mk_dfall['Mann-Kendall_Slope_Coefficient'].max())
colors = plt.cm.Greys(norm(mk_dfall['Mann-Kendall_Slope_Coefficient']))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(4, 4),sharey=True)
ax1.scatter([0] * len(mk_dfall), mk_dfall['Mann-Kendall_Slope_Coefficient'], c=colors, cmap='Greys', s=100, edgecolor='black', linewidth=1.5,zorder=4)
ax1.set_title('MRW_all',fontsize=9)
for i, row in mk_dfall.iterrows():
   ax1.annotate(row['Column'], (0, row['Mann-Kendall_Slope_Coefficient']), textcoords="offset points", xytext=(5,0), ha='left',fontsize=9)
ax1.set_xticks([0],labels='')
ax1.set_ylim(0.008,0.026)
ax1.text(-.105,0.0267,'a)')
#ax1.set_yticks([])
ax1.set_ylabel('1800-2021 Slope',fontsize=9)
#ax1.set_yticklabels([0,0.005,0.01,0.015,0.02,0.025,0.03],fontsize=9)
ax1.grid(True, axis='y')
ax1.axvline(x=0, color='grey', linestyle='--')
ax2.axvline(x=0, color='grey', linestyle='--')
ax2.scatter([0] * len(mk_df300), mk_df300['Mann-Kendall_Slope_Coefficient'], c=colors, cmap='Greys', s=100, edgecolor='black', linewidth=1.5,zorder=4)
ax2.set_title('MRW<200µm',fontsize=9)
for i, row in mk_df300.iterrows():
   ax2.annotate(row['Column'], (0, row['Mann-Kendall_Slope_Coefficient']), textcoords="offset points", xytext=(5,0), ha='left',fontsize=9)
ax2.set_xticks([0],labels='')
ax2.set_ylim(0.008,0.026)
ax2.text(-0.07,0.0267,'b)')
#ax2.set_yticks
#ax2.set_yticklabels([0.008,0.01,0.015,0.02,0.025,0.03],fontsize=9)
ax2.grid(True, axis='y')
plt.tight_layout(w_pad=0)
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', 'pbwmkindustrial300_v2.eps'), format='eps',bbox_inches='tight')
plt.show()

labels=['aMXD10','aMXD20','aMXD40','aMXD80','aMXD100']

#norm = plt.Normalize(sp_df['Spearman_Coefficient'].min(), sp_df['Spearman_Coefficient'].max())
#colors = plt.cm.Greys_r(norm(sp_df['Spearman_Coefficient']))

fig=plt.subplots(1,1,figsize=(2, 4),sharey=True)
plt.axvline(x=0, color='grey', linestyle='--')
plt.scatter([0] * len(sp_df), sp_df['Spearman_Coefficient'], c=colors,cmap='Greys_r', s=100, edgecolor='black', linewidth=1.5,zorder=4)
#lt.title('MRW_all',fontsize=9)
for i, row in sp_df.iterrows():
   plt.annotate(labels[i], (0, row['Spearman_Coefficient']), textcoords="offset points", xytext=(5,0), ha='left',fontsize=9)
plt.xticks([0],labels='')
plt.ylim(0.1,0.5)
#ax1.set_yticks([])
plt.ylabel('Spearman Rank Correlation Coefficient',fontsize=9)
#ax1.set_yticklabels([0,0.005,0.01,0.015,0.02,0.025,0.03],fontsize=9)
plt.grid(True, axis='y')
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', 'spearman_v2.eps'), format='eps',bbox_inches='tight')

