import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import netCDF4 as nc
from datetime import datetime, timedelta
import utils as u  
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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


file_path = '/Users/julieedwards/Documents/Projects/MANCHA/Climate/daily/iera5_t2m_daily_-141.05E_68.67N_n.nc'
dataset = nc.Dataset(file_path)
t2m = dataset.variables['t2m'][:]
time = dataset.variables['time'][:]
time_units = dataset.variables['time'].units
dates = nc.num2date(time, time_units)
dates = [pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates]
df = pd.DataFrame({'date': dates, 'temperature': t2m})

df = df[(df['date'].dt.year >= 1950) & (df['date'].dt.year <= 2021)]

#file_path = '/Users/julieedwards/Documents/Projects/MANCHA/Climate/daily/iberkeley_tavg_daily_full_-141.05E_68.67N_n.nc'
#dataset = nc.Dataset(file_path)
#t2m = dataset.variables['TAVG'][:]
#time = dataset.variables['time'][:]
#time_units = dataset.variables['time'].units
#dates = nc.num2date(time, time_units)
#dates = [pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates]
#df = pd.DataFrame({'date': dates, 'temperature': t2m})
#df = df[(df['date'].dt.year >= 1950) & (df['date'].dt.year <= 2021)]


subsfrcs=sfrcs[['pbw10','pbw20','pbw40','pbw80']]
labelamxd=['aMXD 10 $\mu$m','aMXD 20 $\mu$m','aMXD 40 $\mu$m','aMXD 80 $\mu$m']
panel=['a)','b)','c)','d)']
sub=subsfrcs
fig, axs = plt.subplots(4,1,figsize=(6, 7),sharex=True)


R0 = np.zeros((31, 365))
P0 = np.zeros((31, 365))
for e,column in enumerate(sub.columns):
    for i in range(1, 32):
        df['rolling_mean'] = df['temperature'].rolling(window=i, min_periods=1, center=True).mean()
        for doy in range(1, 366):
            temp_series = df[df['date'].dt.dayofyear == doy][['date', 'rolling_mean']]
            temp_series['year'] = temp_series['date'].dt.year
            temp_series.set_index('year', inplace=True)
            temp_series = temp_series['rolling_mean']
            temp_series = temp_series[temp_series.index.isin(sub.index)]
            if not temp_series.empty:
            # Align with the tree-ring data
                aligned_years = temp_series.index
                aligned_tree_ring = sub.loc[aligned_years][column]
                if len(temp_series) == len(aligned_tree_ring):
                    R, p = pearsonr(temp_series, aligned_tree_ring)
                    R0[i-1, doy-1] = R
                    P0[i-1, doy-1] = p
# Plotting
    vmin, vmax = -.8, .8
    bins = np.linspace(-0.8, 0.8, 17) 
    ax=axs[e]
    x, y = np.meshgrid(np.arange(1, 366), np.arange(1, 32))
    c = ax.contourf(x, y, R0, bins,cmap='coolwarm', extend='both')
    cbar = fig.colorbar(c, ax=ax, orientation='vertical', pad=0.01,
                    ticks=[-0.8,-0.4,0,0.4,0.8],aspect=20,location='right')
    cbar.set_label('R', rotation=0, labelpad=5)
#cbar.ax.tick_params()
#cbar.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_yticks([0,10,20,30])
    stipple_x, stipple_y = np.where(P0 > 0.01)
    ax.scatter(stipple_y + 1, stipple_x,s=.5, c='k', alpha=0.6)
    ax.set_ylabel('Window length',fontsize=9)
#ax.set_title('aMXD (120um resolution)', fontsize=20, y=1.2)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(np.linspace(15, 350, 12))
    ax_top.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax_top.tick_params(axis='x')
    ax_top.set_xlabel(labelamxd[e],fontsize=10)
    ax.text(-12,35,panel[e])
ax.set_xlabel('Day of year (Center of window)',fontsize=10)
plt.tight_layout()
plt.savefig('climatepanel_dailyres.eps', format='eps',bbox_inches='tight')
plt.show()
