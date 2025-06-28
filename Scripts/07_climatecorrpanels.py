import pandas as pd
import numpy as np
import os
import netCDF4 as nc
from datetime import datetime
import utils as u  
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

####### READ ME #########
# This code creates Figure 2 for Edwards et al., 2025? for GRL and supplemental figures
## The aMXD at multiple resolutions is correlated against daily ERA5 temperature data




# READ IN wood anatomy DATA
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

# READ IN ERA5 daily DATA
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Climate', 'iera5_t2m_daily_-141.05E_68.67N_n.nc')
dataset = nc.Dataset(file_path)
t2m = dataset.variables['t2m'][:]
time = dataset.variables['time'][:]
time_units = dataset.variables['time'].units
dates = nc.num2date(time, time_units)
dates = [pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates]
df = pd.DataFrame({'date': dates, 'temperature': t2m})
df = df[(df['date'].dt.year >= 1950) & (df['date'].dt.year <= 2021)]


### Running window correlations for many resolutions
subsfrcs=sfrcs[['pbw10','pbw20','pbw40','pbw80']]
labelamxd=['aMXD 10 $\mu$m','aMXD 20 $\mu$m','aMXD 40 $\mu$m','aMXD 80 $\mu$m']
panel=['a)','b)','c)','d)']
sub=subsfrcs
fig, axs = plt.subplots(4,1,figsize=(6, 7),sharex=True)


R0 = np.zeros((90, 365))
P0 = np.zeros((90, 365))
for e,column in enumerate(sub.columns):
    for i in range(1, 91):
        df['rolling_mean'] = df['temperature'].rolling(window=i, min_periods=1, center=True).mean()
        for doy in range(1, 366):
            temp_series = df[df['date'].dt.dayofyear == doy][['date', 'rolling_mean']]
            temp_series['year'] = temp_series['date'].dt.year
            temp_series.set_index('year', inplace=True)
            temp_series = temp_series['rolling_mean']
            temp_series = temp_series[temp_series.index.isin(sub.index)]
            if not temp_series.empty:
                aligned_years = temp_series.index
                aligned_tree_ring = sub.loc[aligned_years][column]
                if len(temp_series) == len(aligned_tree_ring):
                    R, p = pearsonr(temp_series, aligned_tree_ring)
                    R0[i-1, doy-1] = R
                    P0[i-1, doy-1] = p
# Plotting the correlations with significance stippling
    vmin, vmax = -.8, .8
    bins = np.linspace(-0.8, 0.8, 17) 
    ax=axs[e]
    x, y = np.meshgrid(np.arange(1, 366), np.arange(1, 91))
    c = ax.contourf(x, y, R0, bins,cmap='coolwarm', extend='both')
    cbar = fig.colorbar(c, ax=ax, orientation='vertical', pad=0.01,
                    ticks=[-0.8,-0.4,0,0.4,0.8],aspect=20,location='right')
    cbar.set_label('R', rotation=0, labelpad=5)
    ax.set_yticks([0,20,40,60,80])
    stipple_x, stipple_y = np.where(P0 > 0.01)
    ax.scatter(stipple_y + 1, stipple_x,s=.01, c='k', alpha=0.6)
    ax.set_ylabel('Window length',fontsize=9)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(np.linspace(15, 350, 12))
    ax_top.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax_top.tick_params(axis='x')
    ax_top.set_xlabel(labelamxd[e],fontsize=10)
    ax.text(-35,105,panel[e])
ax.set_xlabel('Day of year (Center of window)',fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', 'climatepanel_ERA5_multires.eps'), format='eps',bbox_inches='tight')
plt.show()


## Running window daily analysis for 'best' chronology aMXD10
x=sfrcs['pbw10']

R0 = np.zeros((90, 365))
P0 = np.zeros((90, 365))
for i in range(1, 91):
    df['rolling_mean'] = df['temperature'].rolling(window=i, min_periods=1, center=True).mean()
    for doy in range(1, 366):
        temp_series = df[df['date'].dt.dayofyear == doy][['date', 'rolling_mean']]
        temp_series['year'] = temp_series['date'].dt.year
        temp_series.set_index('year', inplace=True)
        temp_series = temp_series['rolling_mean']
        temp_series = temp_series[temp_series.index.isin(x.index)]
        if not temp_series.empty:
            # Align with the tree-ring data
            aligned_years = temp_series.index
            aligned_tree_ring = x.loc[aligned_years]
            if len(temp_series) == len(aligned_tree_ring):
                R, p = pearsonr(temp_series, aligned_tree_ring)
                R0[i-1, doy-1] = R
                P0[i-1, doy-1] = p
# find window center and window length with highest R
max_corr = np.max(R0)
max_indices = np.unravel_index(np.argmax(R0, axis=None), R0.shape)
max_window_length = max_indices[0] + 1
max_day_of_year = max_indices[1] + 1

print(f'Maximum correlation coefficient (R0) is {max_corr} for window length {max_window_length} and center at day of year {max_day_of_year}')


######### Load in ERA5 monthly data

###### ERA5 ###################
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Climate', 'monthly_iera5_t2m_-141.05E_68.67N_1950-2022_n.nc')
dataset = nc.Dataset(file_path)
time = dataset.variables['time'][:]
temp = dataset.variables['t2m'][:]
dates = pd.date_range(start='1950-01-15', periods=len(time), freq='M')
era5data = pd.DataFrame({'time': dates, 'tmp': temp})



##### SEASCORR-esque analysis, Meko et al., 2011
#Basically let's us see which seasonal window has the storngest correlation 
df = pd.DataFrame(era5data)
df.set_index('time', inplace=True)
df_monthly = df.resample('M').mean()
df_monthly.reset_index(inplace=True)
df_monthly['year'] = df_monthly['time'].dt.year
df_monthly['month'] = df_monthly['time'].dt.month
df_monthly = df_monthly[['year', 'month', 'tmp']]

df_monthly['prev_tmp'] = df_monthly['tmp'].shift(1)
# For January end month, use December of the previous year etc.. 
# So Jan end month at 3 month seasonal window is november and december of previous year and Jan of current year (ndJ)
for year in df_monthly['year'].unique():
    if not df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 12), 'tmp'].empty:
        df_monthly.loc[(df_monthly['year'] == year) & (df_monthly['month'] == 1), 'prev_tmp'] = df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 12), 'tmp'].values[0]

df_monthly['2_month_avg'] = df_monthly[['tmp', 'prev_tmp']].mean(axis=1)
df_monthly['prev_tmp_1'] = df_monthly['tmp'].shift(1)
df_monthly['prev_tmp_2'] = df_monthly['tmp'].shift(2)
df_monthly['prev_tmp_3'] = df_monthly['tmp'].shift(3)

# For January and February, use December and November of the previous year
for year in df_monthly['year'].unique():
    if not df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 12), 'tmp'].empty:
        prev_dec_temp = df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 12), 'tmp'].values[0]
        df_monthly.loc[(df_monthly['year'] == year) & (df_monthly['month'] == 1), 'prev_tmp_1'] = prev_dec_temp
    
    if not df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 11), 'tmp'].empty:
        prev_nov_temp = df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 11), 'tmp'].values[0]
        df_monthly.loc[(df_monthly['year'] == year) & (df_monthly['month'] == 1), 'prev_tmp_2'] = prev_nov_temp
    if not df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 10), 'tmp'].empty:
        prev_oct_temp = df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 10), 'tmp'].values[0]
        df_monthly.loc[(df_monthly['year'] == year) & (df_monthly['month'] == 1), 'prev_tmp_3'] = prev_oct_temp    
    if not df_monthly.loc[(df_monthly['year'] == year-1) & (df_monthly['month'] == 12), 'tmp'].empty:
        df_monthly.loc[(df_monthly['year'] == year) & (df_monthly['month'] == 2), 'prev_tmp_2'] = prev_dec_temp
    
df_monthly['3_month_avg'] = df_monthly[['tmp', 'prev_tmp_1', 'prev_tmp_2']].mean(axis=1)
df_monthly['4_month_avg'] = df_monthly[['tmp', 'prev_tmp_1', 'prev_tmp_2','prev_tmp_3']].mean(axis=1)
df_monthly_filtered = df_monthly[(df_monthly['year'] >= 1950) & (df_monthly['year'] <= 2021)]

xsfrcs=sfrcs['pbw10']
xsfrcs.index.names = ['year']

df_merged = pd.merge(df_monthly_filtered, xsfrcs, on='year')
correlation_results = {
    'month': [],
    'tmp_corr': [],
    'tmp_pval': [],
    '2_month_avg_corr': [],
    '2_month_avg_pval': [],
    '3_month_avg_corr': [],
    '3_month_avg_pval': [],
    '4_month_avg_corr': [],
    '4_month_avg_pval': []

}
for month in range(1, 13):
    month_data = df_merged[df_merged['month'] == month]
    tmp_corr, tmp_pval = pearsonr(month_data['tmp'], month_data['pbw10'])
    avg2_corr, avg2_pval = pearsonr(month_data['2_month_avg'], month_data['pbw10'])
    avg3_corr, avg3_pval = pearsonr(month_data['3_month_avg'], month_data['pbw10'])   
    avg4_corr, avg4_pval = pearsonr(month_data['4_month_avg'], month_data['pbw10'])   

    correlation_results['month'].append(month)
    correlation_results['tmp_corr'].append(tmp_corr)
    correlation_results['tmp_pval'].append(tmp_pval)
    correlation_results['2_month_avg_corr'].append(avg2_corr)
    correlation_results['2_month_avg_pval'].append(avg2_pval)
    correlation_results['3_month_avg_corr'].append(avg3_corr)
    correlation_results['3_month_avg_pval'].append(avg3_pval)
    correlation_results['4_month_avg_corr'].append(avg4_corr)
    correlation_results['4_month_avg_pval'].append(avg4_pval)    
correlation_df = pd.DataFrame(correlation_results)


# Load the NetCDF file of spatial ERA 5 data
file_path = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Climate', 'ERA_DOY195to223_temperature_field.nc')
dataset = nc.Dataset(file_path, mode='r')
temperature = dataset.variables['t2m'][:]
time = dataset.variables['time'][:]
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
if time[0] > time[-1]:
    time = time[::-1]
    temperature = temperature[::-1, :, :]

#temporal subset on Wood anatomy time series to match length of climate data
annual_time_series = sfrcs.loc[1950:2021]['pbw10']

correlation_map = np.empty_like(temperature[0, :, :])
p_value_map = np.empty_like(temperature[0, :, :])

# Loop over latitudes and longitudes to calculate R for each grid, there's probably a better/faster way to do this
for i in range(temperature.shape[1]):
    for j in range(temperature.shape[2]):
        correlation_map[i, j], p_value_map[i, j] = pearsonr(temperature[:, i, j], annual_time_series)

# Adjust longitudes to range from -180 to 180
lon = np.where(lon > 180, lon - 360, lon)
sorted_indices = np.argsort(lon)
lon_sorted = lon[sorted_indices]
correlation_map_sorted = correlation_map[:, sorted_indices]
p_value_map_sorted = p_value_map[:, sorted_indices]

# Subset the data for the specified lat/lon range
lat_indices = np.where((lat >= 40) & (lat <= 90))[0]
lon_indices = np.where((lon_sorted >= -180) & (lon_sorted <= -1))[0]
lat_subset = lat[lat_indices]
lon_subset = lon_sorted[lon_indices]
# subset correlaiton results
correlation_subset = correlation_map_sorted[np.ix_(lat_indices, lon_indices)]
p_value_subset = p_value_map_sorted[np.ix_(lat_indices, lon_indices)]



####### PLOTTING ########



fig = plt.figure(figsize=(5, 5.5))
#### PANEL A
gs = fig.add_gridspec(1, 4,left=0.0, right=1,top=1,bottom=.8,
                        wspace=0,hspace=0)

ax1 = fig.add_subplot(gs[0, :])

vmin, vmax = -.8, .8
bins = np.linspace(-0.8, 0.8, 17)
x, y = np.meshgrid(np.arange(1, 366), np.arange(1, 91))
c = ax1.contourf(x, y, R0, bins, cmap='coolwarm', extend='both')
cbar = fig.colorbar(c,  orientation='vertical', pad=0.01,
                    ticks=[-0.8, -0.4, 0, 0.4, 0.8], aspect=20, location='right')
cbar.set_label('R', rotation=0, labelpad=5,fontsize=9)
cbar.set_ticklabels(ticklabels=[-0.8,-0.4,0,0.4,0.8],fontsize=9)
stipple_x, stipple_y = np.where(P0 > 0.01)
ax1.scatter(stipple_y + 1, stipple_x, s=.01, c='k', alpha=0.6)
ax1.yaxis.set_label_position("left")
ax1.yaxis.tick_left()
ax1.set_xticks([50,100,150,200,250,300,350])
ax1.set_xticklabels(labels=[50,100,150,200,250,300,350],fontsize=9)
ax1.set_yticks([0,20,40,60,80])
ax1.set_yticklabels(labels=[0,20,40,60,80],fontsize=9)
ax1.set_ylabel('Window length (days)', fontsize=9)
ax1.set_xlabel('Day of year (Center of window)', fontsize=9)
ax_top = ax1.twiny()
ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks(np.linspace(15, 350, 12))
ax_top.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],fontsize=9)
ax_top.tick_params(axis='x')
ax_top.set_xlabel('Month', fontsize=9)
fig.text(-.09, 1.05, 'a)', fontsize=9)


### PANEL B
gs1 = fig.add_gridspec(1, 4,left=-.05, right=.84, top=.65,bottom=.45,
                        wspace=0,hspace=0)
# Plot b)
axs = [fig.add_subplot(gs1[0, i]) for i in range(4)]
ylim = (-.2, .8)
months_labels = ['J', 'M', 'M', 'J', 'S', 'N']

def get_bar_colors(pvals):
    return ['#2367AD' if p < 0.01 else '#D6EAF4' for p in pvals]

# Plotting tmp correlation
colors = get_bar_colors(correlation_df['tmp_pval'])
axs[0].bar(correlation_df['month'], correlation_df['tmp_corr'], color=colors)
axs[0].set_title('1-Month', fontsize=9)
axs[0].set_xlabel('End Month', fontsize=9)
axs[0].set_ylim(ylim)
axs[0].set_xticks([1, 3, 5, 7, 9, 11])
axs[0].set_xticklabels(months_labels, fontsize=10)
axs[0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, zorder=0)
axs[0].set_axisbelow(True)
axs[0].tick_params(axis='both', direction='in', labelsize=9)
axs[0].set_yticks([-.2,0,.2,.4,.6,.8])
axs[0].set_yticks([-.2,0,.2,.4,.6,.8])
axs[0].set_yticklabels(labels=[],fontsize=9)

# Plotting 2-month average correlation
colors = get_bar_colors(correlation_df['2_month_avg_pval'])
axs[1].bar(correlation_df['month'], correlation_df['2_month_avg_corr'], color=colors)
axs[1].set_title('2-Month', fontsize=9)
axs[1].set_xlabel('End Month', fontsize=9)
axs[1].set_ylim(ylim)
axs[1].set_xticks([1, 3, 5, 7, 9, 11])
axs[1].set_xticklabels(months_labels, fontsize=9)
axs[1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, zorder=0)
axs[1].set_axisbelow(True)
axs[1].tick_params(axis='both', direction='in', labelsize=9)
axs[1].set_yticks([-.2,0,.2,.4,.6,.8])

# Plotting 3-month average correlation
colors = get_bar_colors(correlation_df['3_month_avg_pval'])
axs[2].grid(linestyle='--', zorder=1)
axs[2].bar(correlation_df['month'], correlation_df['3_month_avg_corr'], color=colors)
axs[2].set_title('3-Month', fontsize=9)
axs[2].set_xlabel('End Month', fontsize=9)
axs[2].set_ylim(ylim)
axs[2].set_xticks([1, 3, 5, 7, 9, 11])
axs[2].set_xticklabels(months_labels, fontsize=9)
axs[2].grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, zorder=0)
axs[2].set_axisbelow(True)
axs[2].tick_params(axis='both', direction='in', labelsize=9)
axs[2].set_yticks([-.2,0,.2,.4,.6,.8])

colors = get_bar_colors(correlation_df['4_month_avg_pval'])
axs[3].grid(linestyle='--', zorder=1)
axs[3].bar(correlation_df['month'], correlation_df['4_month_avg_corr'], color=colors)
axs[3].set_title('4-Month', fontsize=9)
axs[3].set_xlabel('End Month', fontsize=9)
axs[3].set_ylim(ylim)
axs[3].set_xticks([1, 3, 5, 7, 9, 11])
axs[3].set_xticklabels(months_labels, fontsize=9)
axs[3].grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, zorder=0)
axs[3].set_axisbelow(True)
axs[3].set_ylabel('R', fontsize=9, rotation=0)
axs[3].yaxis.set_label_position("right")
axs[3].yaxis.tick_right()
axs[3].tick_params(axis='y', direction='in', labelright=True, labelleft=False, labelsize=9)
axs[3].set_yticks([-.2,0,.2,.4,.6,.8])
axs[3].set_yticklabels(labels=[-0.2,0,0.2,0.4,0.6,0.8],fontsize=9)

# Ensure y-axis sharing manually
for ax in axs[:3]:
    ax.tick_params(axis='y', labelleft=False)

blue_patch = plt.Line2D([0], [0], color='#2367AD', lw=3, label='p < 0.01')
lightblue_patch = plt.Line2D([0], [0], color='#D6EAF4', lw=3, label='p ≥ 0.01')
legend = axs[0].legend(handles=[blue_patch, lightblue_patch], loc='upper left', fontsize=6,handlelength=0.5)
legend.get_frame().set_linewidth(0)
fig.text(-.09, 0.65, 'b)', fontsize=9)


########### PANEL C
gs2 = fig.add_gridspec(1, 4,left=-.0, right=.86, top=0.45,bottom=0,
                        wspace=0,hspace=0)
# Plot c)
ax3 = fig.add_subplot(gs2[0, :], projection=ccrs.PlateCarree())  
ax3.set_extent([-180, -70, 50, 80], crs=ccrs.PlateCarree())
ax3.coastlines()
lon_grid, lat_grid = np.meshgrid(lon_subset, lat_subset)
correlation_plot = ax3.contourf(lon_grid, lat_grid, correlation_subset, bins, cmap='coolwarm', extend='both', transform=ccrs.PlateCarree())
stipple_mask = p_value_subset > 0.01
stipple_indices = np.where(stipple_mask)
downsample_rate = 1  
downsampled_indices = (stipple_indices[0][::downsample_rate], stipple_indices[1][::downsample_rate])
ax3.scatter(lon_grid[downsampled_indices], lat_grid[downsampled_indices], s=.005, c='k', alpha=1, transform=ccrs.PlateCarree())  # Ensure correct use of ax3

cbar = fig.colorbar(correlation_plot, ax=ax3, orientation='vertical', fraction=0.0133, pad=0.01,
                    ticks=[-0.8, -0.4, 0, 0.4, 0.8], aspect=20, location='right')
cbar.set_label('R', rotation=0, labelpad=5,fontsize=9)
cbar.set_ticklabels(ticklabels=[-0.8, -0.4, 0, 0.4, 0.8],fontsize=9)
lat_point = 68.67
lon_point = -141.05
ax3.plot(lon_point, lat_point, 'o', markersize=5, color='k', transform=ccrs.PlateCarree())
ax3.text(lon_point - 4, lat_point + 2.5, 'Firth River', transform=ccrs.PlateCarree(),fontsize=9)
ax3.add_feature(cfeature.BORDERS)
ax3.add_feature(cfeature.LAND)
ax3.add_feature(cfeature.OCEAN)
gl = ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.5, color='gray')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}
plt.title('ERA5 195 to 223 DOY mean', fontsize=9)
fig.text(-0.09, 0.32, 'c)', fontsize=9)
plt.subplots_adjust(left=0.0, right=1, top=0.95, bottom=0.05, hspace=0)


plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures', 'climatepanel_ERA5.eps'), format='eps',bbox_inches='tight')
plt.show()
