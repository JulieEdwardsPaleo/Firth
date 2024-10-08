import os
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
import numpy as np
from pymannkendall import original_test as mk_test
import os
import pandas as pd

def parse_filename(filename):
    parts = filename.split('_')
    if len(parts) < 2 or not parts[1].startswith('stats'):
        raise ValueError(f"Unexpected filename format: {filename}")

    power_transformed = 'p' in parts[1]
    detrend_method = 'rcs' if 'rcs' in parts[1] else 'sf-rcs'
    aggregation_method = 'q' if 'q' in parts[1] else 'bw'
    
    # Extract resolution
    resolution_str = ''.join(filter(str.isdigit, parts[1]))
    if resolution_str:
        resolution = int(resolution_str)
    else:
        raise ValueError(f"Resolution not found in filename: {filename}")
    
    return {
        'power_transformed': power_transformed,
        'detrend_method': detrend_method,
        'aggregation_method': aggregation_method,
        'resolution': resolution
    }

def extract_rbar_eff(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'rbar.eff' in df.columns:
            return df['rbar.eff'].iloc[0]
        else:
            return None
    except Exception as e:
        return None

def main(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith('RWI_stats'):
            filepath = os.path.join(folder_path, filename)
            try:
                params = parse_filename(filename)
                rbar_eff = extract_rbar_eff(filepath)
                if rbar_eff is not None:
                    params['rbar.eff'] = rbar_eff
                    results.append(params)
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(results)
    return df

def kruskal_wallis_test(df, group_col):
    groups = [df[df[group_col] == val]['rbar.eff'] for val in df[group_col].unique()]
    stat, p_value = kruskal(*groups)
    return stat, p_value

def sensitivity_analysis(df):
    analysis_results = {}

    # Analysis for power transformation
    power_transformed_group = df.groupby('power_transformed')['rbar.eff'].mean().reset_index()
    power_transformed_group['stat'], power_transformed_group['p_value'] = kruskal_wallis_test(df, 'power_transformed')
    analysis_results['power_transformed'] = power_transformed_group

    # Analysis for detrend method
    detrend_method_group = df.groupby('detrend_method')['rbar.eff'].mean().reset_index()
    detrend_method_group['stat'], detrend_method_group['p_value'] = kruskal_wallis_test(df, 'detrend_method')
    analysis_results['detrend_method'] = detrend_method_group

    # Analysis for aggregation method
    aggregation_method_group = df.groupby('aggregation_method')['rbar.eff'].mean().reset_index()
    aggregation_method_group['stat'], aggregation_method_group['p_value'] = kruskal_wallis_test(df, 'aggregation_method')
    analysis_results['aggregation_method'] = aggregation_method_group

    # Analysis for resolution
    resolution_group = df.groupby('resolution')['rbar.eff'].mean().reset_index()
    resolution_group['stat'], resolution_group['p_value'] = kruskal_wallis_test(df, 'resolution')
    analysis_results['resolution'] = resolution_group

    return analysis_results

# Define the folder path
folder_path = '/Users/julieedwards/Documents/Projects/MANCHA/MXD/nomcrb08_June2024'

# Get the DataFrame with the extracted data
df = main(folder_path)

# Perform sensitivity analysis
sensitivity_results = sensitivity_analysis(df)


def parse_filename(filename):
    parts = filename.split('_')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")

    stats_part = parts[1]

    power_transformed = 'p' in stats_part
    detrend_method = 'rcs' if 'rcs' in stats_part else 'sf-rcs'
    aggregation_method = 'q' if 'q' in stats_part else 'bw'
    
    # Extract resolution
    resolution_str = ''.join(filter(str.isdigit, stats_part))
    if resolution_str:
        resolution = int(resolution_str)
    else:
        raise ValueError(f"Resolution not found in filename: {filename}")
    
    return {
        'power_transformed': power_transformed,
        'detrend_method': detrend_method,
        'aggregation_method': aggregation_method,
        'resolution': resolution
    }

def extract_rbar_eff(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'rbar.eff' in df.columns:
            return df['rbar.eff'].iloc[0]
        else:
            return None
    except Exception as e:
        return None

def extract_std_values(filepath):
    try:
        df = pd.read_csv(filepath)
        df.index=df['Unnamed: 0']
        return df['std']
    except Exception as e:
        return None

def calculate_lag1_autocorrelation(data):
    return data.autocorr(lag=1)

def calculate_mann_kendall_slope(data):
    result = mk_test(data)
    return result.slope

def main(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith('noFill'):
            filepath = os.path.join(folder_path, filename)
            try:
                params = parse_filename(filename)
                std_values = extract_std_values(filepath)
                if std_values is not None:
                    # Select values from 1150 to 2021
                    std_values_filtered = std_values[std_values.index.isin(range(1150, 2022))]
                    if not std_values_filtered.empty:
                        ar1 = calculate_lag1_autocorrelation(std_values_filtered)
                        mk_slope = calculate_mann_kendall_slope(std_values_filtered)
                        params['ar1'] = ar1
                        params['mk_slope'] = mk_slope
                        params['mean'] = np.mean(std_values_filtered)
                        params['std'] = np.std(std_values_filtered)

                rbar_eff = extract_rbar_eff(filepath)
                if rbar_eff is not None:
                    params['rbar.eff'] = rbar_eff
                if 'ar1' in params or 'rbar.eff' in params:
                    results.append(params)
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(results)
    return df

def kruskal_wallis_test(df, group_col, value_col):
    groups = [df[df[group_col] == val][value_col] for val in df[group_col].unique()]
    stat, p_value = kruskal(*groups)
    return stat, p_value

def sensitivity_analysis(df, value_col):
    analysis_results = {}

    # Ensure all expected columns are present
    expected_cols = ['power_transformed', 'detrend_method', 'aggregation_method', 'resolution']
    for col in expected_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column {col} not found in DataFrame.")

    # Analysis for power transformation
    power_transformed_group = df.groupby('power_transformed').agg({value_col: 'mean'}).reset_index()
    stat, p_value = kruskal_wallis_test(df, 'power_transformed', value_col)
    power_transformed_group['stat'] = stat
    power_transformed_group['p_value'] = p_value
    analysis_results['power_transformed'] = power_transformed_group

    # Analysis for detrend method
    detrend_method_group = df.groupby('detrend_method').agg({value_col: 'mean'}).reset_index()
    stat, p_value = kruskal_wallis_test(df, 'detrend_method', value_col)
    detrend_method_group['stat'] = stat
    detrend_method_group['p_value'] = p_value
    analysis_results['detrend_method'] = detrend_method_group

    # Analysis for aggregation method
    aggregation_method_group = df.groupby('aggregation_method').agg({value_col: 'mean'}).reset_index()
    stat, p_value = kruskal_wallis_test(df, 'aggregation_method', value_col)
    aggregation_method_group['stat'] = stat
    aggregation_method_group['p_value'] = p_value
    analysis_results['aggregation_method'] = aggregation_method_group

    # Analysis for resolution
    resolution_group = df.groupby('resolution').agg({value_col: 'mean'}).reset_index()
    stat, p_value = kruskal_wallis_test(df, 'resolution', value_col)
    resolution_group['stat'] = stat
    resolution_group['p_value'] = p_value
    analysis_results['resolution'] = resolution_group

    return analysis_results

# Define the folder path
folder_path = '/Users/julieedwards/Documents/Projects/MANCHA/MXD/nomcrb08_June2024'

# Get the DataFrame with the extracted data
df = main(folder_path)


ar1_results = sensitivity_analysis(df, 'ar1')
mk_slope_results = sensitivity_analysis(df, 'mk_slope')
mean_results = sensitivity_analysis(df, 'mean')
std_results = sensitivity_analysis(df, 'std')


print(sensitivity_results)
print(ar1_results)
print(mk_slope_results)
print(mean_results)
print(std_results)
