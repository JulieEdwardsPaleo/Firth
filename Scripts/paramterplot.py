import pandas as pd
import matplotlib.pyplot as plt

# Data provided by the user
data = {
    'power_transformed': {
        'rbar.eff': pd.DataFrame({
            'power_transformed': [False, True],
            'rbar.eff': [0.34745, 0.34680],
            'p_value': [0.817914, 0.817914]
        }),
        'ar1': pd.DataFrame({
            'power_transformed': [False, True],
            'ar1': [0.129564, 0.145333],
            'stat': [236.0, 236.0],
            'p_value': [0.330154, 0.330154]
        }),
        'mk_slope': pd.DataFrame({
            'power_transformed': [False, True],
            'mk_slope': [0.000034, 0.000035],
            'stat': [217.0, 217.0],
            'p_value': [0.645623, 0.645623]
        }),
        'mean': pd.DataFrame({
            'power_transformed': [False, True],
            'mean': [1.000089, 0.998497],
            'stat': [136.0, 136.0],
            'p_value': [0.083415, 0.083415]
        }),
        'std': pd.DataFrame({
            'power_transformed': [False, True],
            'std': [0.052047, 0.050661],
            'stat': [177.0, 177.0],
            'p_value': [0.533842, 0.533842]
        })
    },
    'detrend_method': {
        'rbar.eff': pd.DataFrame({
            'detrend_method': ['rcs', 'sf-rcs'],
            'rbar.eff': [0.34710, 0.34715],
            'p_value': [0.946013, 0.946013]
        }),
        'ar1': pd.DataFrame({
            'detrend_method': ['rcs', 'sf-rcs'],
            'ar1': [0.124120, 0.150778],
            'p_value': [0.051462, 0.051462]
        }),
        'mk_slope': pd.DataFrame({
            'detrend_method': ['rcs', 'sf-rcs'],
            'mk_slope': [0.000031, 0.000038],
            'p_value': [0.078704, 0.078704]
        }),
        'mean': pd.DataFrame({
            'detrend_method': ['rcs', 'sf-rcs'],
            'mean': [1.000181, 0.998406],
            'p_value': [0.02149, 0.02149]
        }),
        'std': pd.DataFrame({
            'detrend_method': ['rcs', 'sf-rcs'],
            'std': [0.051845, 0.050862],
            'p_value': [0.807656, 0.807656]
        })
    },
    'aggregation_method': {
        'rbar.eff': pd.DataFrame({
            'aggregation_method': ['bw', 'q'],
            'rbar.eff': [0.33870, 0.35555],
            'p_value': [0.515656, 0.515656]
        }),
        'ar1': pd.DataFrame({
            'aggregation_method': ['bw', 'q'],
            'ar1': [0.164072, 0.110825],
            'p_value': [0.000356, 0.000356]
        }),
        'mk_slope': pd.DataFrame({
            'aggregation_method': ['bw', 'q'],
            'mk_slope': [0.000036, 0.000033],
            'p_value': [0.255913, 0.255913]
        }),
        'mean': pd.DataFrame({
            'aggregation_method': ['bw', 'q'],
            'mean': [0.997805, 1.000781],
            'p_value': [0.000592, 0.000592]
        }),
        'std': pd.DataFrame({
            'aggregation_method': ['bw', 'q'],
            'std': [0.055768, 0.046940],
            'p_value': [0.001552, 0.001552]
        })
    },
    'resolution': {
        'rbar.eff': pd.DataFrame({
            'resolution': [10, 20, 40, 80, 100],
            'rbar.eff': [0.324250, 0.435500, 0.381000, 0.307250, 0.287625],
            'stat': [28.641084] * 5,
            'p_value': [0.000009] * 5
        }),
        'ar1': pd.DataFrame({
            'resolution': [10, 20, 40, 80, 100],
            'ar1': [0.162232, 0.095567, 0.125541, 0.155694, 0.148209],
            'stat': [9.62561] * 5,
            'p_value': [0.047229] * 5
        }),
        'mk_slope': pd.DataFrame({
            'resolution': [10, 20, 40, 80, 100],
            'mk_slope': [0.000039, 0.000042, 0.000041, 0.000029, 0.000022],
            'stat': [24.879878] * 5,
            'p_value': [0.000053] * 5
        }),
        'mean': pd.DataFrame({
            'resolution': [10, 20, 40, 80, 100],
            'mean': [1.001136, 1.000955, 0.999473, 0.997775, 0.997128],
            'stat': [13.688415] * 5,
            'p_value': [0.008359] * 5
        }),
        'std': pd.DataFrame({
            'resolution': [10, 20, 40, 80, 100],
            'std': [0.036318, 0.048755, 0.055846, 0.058399, 0.057450],
            'stat': [23.943293] * 5,
            'p_value': [0.000082] * 5
        })
    }
}



# Extracting unique p-values for each variable and parameter
unique_p_values = []

for category, metrics in data.items():
    for metric, df in metrics.items():
        for p_value in df['p_value'].unique():
            unique_p_values.append({
                'category': category,
                'metric': metric,
                'p_value': p_value
            })

unique_p_values_df = pd.DataFrame(unique_p_values)
import seaborn as sns

# Plotting unique p-values with color representing the category
# Plotting unique p-values with color representing the category and labeled metric ticks
plt.figure(figsize=(4, 4))

palette = sns.color_palette("deep", n_colors=4)

for idx, category in enumerate(unique_p_values_df['category'].unique()):
    subset = unique_p_values_df[unique_p_values_df['category'] == category]
    plt.scatter(subset['metric'], subset['p_value'], label=category, color=palette[idx])

plt.title('Kruskal–Wallis Test Results')
plt.xlabel('')
plt.ylabel('p-value (log scale)')
plt.yscale('log')  # Set y-axis to logarithmic scale

# Add horizontal lines for significance levels
plt.axhline(y=0.01, color='k', linestyle='--', linewidth=1.5, label=r'$\alpha = 0.01$')
plt.axhline(y=0.05, color='red', linestyle='--', label=r'$\alpha = 0.05$')

# Place legend outside the plot
plt.legend(frameon=False, fontsize=8, handlelength=1, borderpad=0,
           ncols=1, columnspacing=0.5, labelspacing=0.5, handletextpad=0.5,
           bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust as needed

plt.xticks(ticks=range(len(unique_p_values_df['metric'].unique())), labels=['rbar', 'ar1', 'slope', 'mean', 'SD'], rotation=45)
plt.grid(linewidth=0.5, linestyle='--')
plt.savefig('parameter_test.eps', format='eps', bbox_inches='tight')
plt.show()
