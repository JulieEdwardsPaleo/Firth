
import pandas as pd
import numpy as np 

# LLW - add to toGIT folder
LLWfile = '/Users/julieedwards/Documents/Projects/MANCHA/QWAData/FINAL/concat/Summary/LWW_biweight_10mu.txt'
df = pd.read_csv(LLWfile, delim_whitespace=True)  # adjust if it's comma-delimited

# Drop MCRB08B
df = df.drop(columns='MCRB_08B', errors='ignore')
column_map = {col.replace('_', ''): col for col in df.columns if col != 'YEAR'}

exclude = pd.read_csv("/Users/julieedwards/Documents/Projects/MANCHA/toGit/Firth/Data/QWA/raw/NA_interpolations.csv")

for _, row in exclude.iterrows():
    year = row['Year']
    raw_series = row['SeriesID']
    col = column_map.get(raw_series)
    if col:
        df.loc[df['YEAR'] == year, col] = pd.NA


df_no_year = df.drop(columns='YEAR')

all_lww = df_no_year.to_numpy().flatten()
all_lww = all_lww[~pd.isna(all_lww)]

# Compute mean and standard deviation
mean_lww = np.mean(all_lww)
sd_lww = np.std(all_lww, ddof=1)  # sample SD

print(f"Mean LWW: {mean_lww:.2f} µm ± {sd_lww:.2f}")
