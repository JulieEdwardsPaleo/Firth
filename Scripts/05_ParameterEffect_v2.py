import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal
from pymannkendall import original_test as mk_test
import matplotlib.pyplot as plt
import seaborn as sns

# reading in data for rbar output from 2_noFillchronos_FINAL
def parse_filename_stats(filename):
    """Parse filenames starting with 'RWI_stats'."""
    parts = filename.split('_')
    if len(parts) < 2 or not parts[1].startswith('stats'):
        raise ValueError(f"Unexpected filename format: {filename}")
#input files have to follow existing naming convention
    power_transformed = 'p' in parts[1]
    detrend_method = 'rcs' if 'rcs' in parts[1] else 'sf-rcs' #no rcs prefix in filename means sf-rcs detrending
    aggregation_method = 'q' if 'q' in parts[1] else 'bw'
    resolution_str = ''.join(filter(str.isdigit, parts[1]))
    if resolution_str:
        resolution = int(resolution_str)
    return {
        'power_transformed': power_transformed,
        'detrend_method': detrend_method,
        'aggregation_method': aggregation_method,
        'resolution': resolution
    }
# reading in data for other chronology statistics output from chronology directory
def parse_filename_chronologies(filename):
    """Parse filenames starting with 'noFill'."""
    parts = filename.split('_')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")
# could probably combine these two functions for the two cases, filenames startign with RWI or noFill
    power_transformed = 'p' in parts[1]
    detrend_method = 'rcs' if 'rcs' in parts[1] else 'sf-rcs'
    aggregation_method = 'q' if 'q' in parts[1] else 'bw'
    resolution_str = ''.join(filter(str.isdigit, parts[1]))
    if resolution_str:
        resolution = int(resolution_str)
    else:
        resolution = np.nan  
    return {
        'power_transformed': power_transformed,
        'detrend_method': detrend_method,
        'aggregation_method': aggregation_method,
        'resolution': resolution
    }

# extracting chronology metrics or calculating them
def extract_rbar_eff(filepath):
    try:
        df = pd.read_csv(filepath)
        return df['rbar.eff'].iloc[0] if 'rbar.eff' in df.columns else None
    except Exception:
        return None

def extract_std_values(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'Unnamed: 0' in df.columns:
            df.index = df['Unnamed: 0']
        return df['std'] if 'std' in df.columns else None
    except Exception:
        return None

def calculate_lag1_autocorrelation(data):
    return data.autocorr(lag=1)

def calculate_mann_kendall_slope(data):
    result = mk_test(data)
    return result.slope

#main function to tie it all together
def main(folder_path, is_chronology_stats=True):
    results = []
    for filename in os.listdir(folder_path):
        if (is_chronology_stats and filename.startswith('RWI_stats')) or (not is_chronology_stats and filename.startswith('noFill')):
            filepath = os.path.join(folder_path, filename)
            
            params = parse_filename_stats(filename) if is_chronology_stats else parse_filename_chronologies(filename)
                
            if is_chronology_stats:
                rbar_eff = extract_rbar_eff(filepath)
                if rbar_eff is not None:
                    params['rbar.eff'] = rbar_eff
                    results.append(params)
            else:
                std_values = extract_std_values(filepath)
                if std_values is not None:
                    # Select values from 1150 to 2021, sorry this is hard-coded
                    std_values_filtered = std_values[std_values.index.isin(range(1150, 2022))]
                    if not std_values_filtered.empty:
                        params['ar1'] = calculate_lag1_autocorrelation(std_values_filtered)
                        params['mk_slope'] = calculate_mann_kendall_slope(std_values_filtered)
                        params['mean'] = np.mean(std_values_filtered)
                        params['std'] = np.std(std_values_filtered)
                rbar_eff = extract_rbar_eff(filepath)
                if rbar_eff is not None:
                    params['rbar.eff'] = rbar_eff
                # Only append if at least one of the new metrics is present
                if any(k in params for k in ['ar1', 'mk_slope', 'mean', 'std', 'rbar.eff']):
                    results.append(params)
                continue 
    return pd.DataFrame(results)

def kruskal_wallis_test(df, group_col, value_col):
    groups = [df[df[group_col] == val][value_col] for val in df[group_col].unique()]
    return kruskal(*groups)

def sensitivity_analysis(df, value_col, group_cols):
    analysis_results = {}
    for col in group_cols:
        # Drop NaN values for the group column and value column
        df_clean = df.dropna(subset=[col, value_col])
        group = df_clean.groupby(col).agg({value_col: 'mean'}).reset_index()
        stat, p_value = kruskal_wallis_test(df_clean, col, value_col)
        group['stat'] = stat
        group['p_value'] = p_value
        analysis_results[col] = group
    return analysis_results

# Define the folder paths
folder_path_stats = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronology_stats')
folder_path_chronologies = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'QWA', 'chronologies')

# Get the DataFrames with the read-in data
df_stats = main(folder_path_stats, is_chronology_stats=True)
df_chronologies = main(folder_path_chronologies, is_chronology_stats=False)

# Perform sensitivity analysis
# For rbar
group_cols_stats = ['power_transformed', 'detrend_method', 'aggregation_method', 'resolution']
rbar_sensitivity_results = sensitivity_analysis(df_stats, 'rbar.eff', group_cols_stats)

# For chronology stats
group_cols_chronologies = ['power_transformed', 'detrend_method', 'aggregation_method', 'resolution']
ar1_results = sensitivity_analysis(df_chronologies, 'ar1', group_cols_chronologies)
mk_slope_results = sensitivity_analysis(df_chronologies, 'mk_slope', group_cols_chronologies)
mean_results = sensitivity_analysis(df_chronologies, 'mean', group_cols_chronologies)
std_results = sensitivity_analysis(df_chronologies, 'std', group_cols_chronologies)

# Prepare data for plotting
data = {
    'power_transformed': {
        'rbar.eff': rbar_sensitivity_results.get('power_transformed'),
        'ar1': ar1_results.get('power_transformed'),
        'mk_slope': mk_slope_results.get('power_transformed'),
        'mean': mean_results.get('power_transformed'),
        'std': std_results.get('power_transformed')
    },
    'detrend_method': {
        'rbar.eff': rbar_sensitivity_results.get('detrend_method'),
        'ar1': ar1_results.get('detrend_method'),
        'mk_slope': mk_slope_results.get('detrend_method'),
        'mean': mean_results.get('detrend_method'),
        'std': std_results.get('detrend_method')
    },
    'aggregation_method': {
        'rbar.eff': rbar_sensitivity_results.get('aggregation_method'),
        'ar1': ar1_results.get('aggregation_method'),
        'mk_slope': mk_slope_results.get('aggregation_method'),
        'mean': mean_results.get('aggregation_method'),
        'std': std_results.get('aggregation_method')
    },
    'resolution': {
        'rbar.eff': rbar_sensitivity_results.get('resolution'),
        'ar1': ar1_results.get('resolution'),
        'mk_slope': mk_slope_results.get('resolution'),
        'mean': mean_results.get('resolution'),
        'std': std_results.get('resolution')
    }
}

#plotting function
def plot_unique_p_values(data):
    unique_p_values = []
    for category, metrics in data.items():
        for metric, df in metrics.items():
            for _, row in df.iterrows():
                unique_p_values.append({
                    'category': category,
                    'metric': metric,
                    'p_value': row['p_value']
                })
    
    unique_p_values_df = pd.DataFrame(unique_p_values)
    plt.figure(figsize=(5.5, 4))    
    palette = sns.color_palette("deep", n_colors=4)
    
    for idx, category in enumerate(unique_p_values_df['category'].unique()):
        subset = unique_p_values_df[unique_p_values_df['category'] == category]
        plt.scatter(subset['metric'], subset['p_value'], label=category, color=palette[idx])
    
    plt.title('Kruskal–Wallis Test Results',fontsize=9)
    plt.ylabel('p-value (log scale)',fontsize=9)
    plt.yscale('log')
    plt.axhline(y=0.01, color='k', linestyle='--', linewidth=1.5, label=r'$\alpha = 0.01$')
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label=r'$\alpha = 0.05$')
    
    # Place legend outside the plot
    plt.legend(frameon=False, fontsize=8, handlelength=1, borderpad=0,
               ncols=1, columnspacing=0.5, labelspacing=0.5, handletextpad=0.5,
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust x-ticks labels
    metrics_unique = ['rbar.eff', 'ar1', 'mk_slope', 'mean', 'std']
    plt.xticks(ticks=range(len(metrics_unique)), labels=['rbar', 'AR1', 'Slope', 'Mean', 'SD'], rotation=45,fontsize=9)
    plt.grid(linewidth=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.getcwd()), 'Figures','parameter_test.eps'), format='eps', bbox_inches='tight')
    plt.show()


# Plot the results, saving to Figures directory
plot_unique_p_values(data)

