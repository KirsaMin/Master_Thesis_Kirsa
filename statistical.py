import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import os

# Define the file paths for the datasets
datasets = {
    'step_time_left': 'data/features/step_time_left.csv',
    'step_time_right': 'data/features/step_time_right.csv',
    'stance_time_left': 'data/features/stance_time_left.csv',
    'stance_time_right': 'data/features/stance_time_right.csv',
    'swing_time_left': 'data/features/swing_time_left.csv',
    'swing_time_right': 'data/features/swing_time_right.csv',
    'stride_time_left': 'data/features/stride_time_left.csv',
    'stride_time_right': 'data/features/stride_time_right.csv',
    'double_support_time': 'data/features/double_support_time.csv'
}

# Create the directory to store the statistical results if it doesn't exist
os.makedirs('./data/statistical', exist_ok=True)

# Perform Kruskal-Wallis test for each dataset
results = []
for name, filepath in datasets.items():
    df = pd.read_csv(filepath)
    ataxia = df[df['label'] == 'EOA'][name]
    dcd = df[df['label'] == 'DCD'][name]
    control = df[df['label'] == 'Control'][name]
    n_ataxia = len(ataxia)
    n_dcd = len(dcd)
    n_control = len(control)

    # Calculate the mean ranks for each group
    combined_ranks = pd.concat([ataxia.rank(), dcd.rank(), control.rank()])
    mean_ranks = [combined_ranks.loc[ataxia.index].mean(), combined_ranks.loc[dcd.index].mean(),
                  combined_ranks.loc[control.index].mean()]

    # Calculate the mean, standard deviation, and perform the Kruskal-Wallis test
    mean_ataxia = ataxia.mean()
    mean_dcd = dcd.mean()
    mean_control = control.mean()
    std_ataxia = ataxia.std()
    std_dcd = dcd.std()
    std_control = control.std()
    result = kruskal(ataxia, dcd, control)

    # Store the results in a dictionary
    results.append({
        'feature': name,
        'pvalue': result.pvalue,
        'chi_square': result.statistic,
        'n_ataxia': n_ataxia,
        'n_dcd': n_dcd,
        'n_control': n_control,
        'mean_rank_ataxia': mean_ranks[0],
        'mean_rank_dcd': mean_ranks[1],
        'mean_rank_control': mean_ranks[2],
        'mean_ataxia': mean_ataxia,
        'mean_dcd': mean_dcd,
        'mean_control': mean_control,
        'std_ataxia': std_ataxia,
        'std_dcd': std_dcd,
        'std_control': std_control
    })

# Convert the results to a DataFrame and save it to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('./data/statistical/kw_results.csv', index=False)

# Perform Mann-Whitney U test: DCD vs Control
results_mw1 = []
for name, filepath in datasets.items():
    df = pd.read_csv(filepath)
    dcd = df[df['label'] == 'DCD'][name]
    control = df[df['label'] == 'Control'][name]
    n_dcd = len(dcd)
    n_control = len(control)

    # Calculate the mean ranks for each group
    combined_ranks = pd.concat([dcd.rank(), control.rank()])
    mean_ranks = [combined_ranks.loc[dcd.index].mean(), combined_ranks.loc[control.index].mean()]

    # Calculate the mean, standard deviation, and perform the Mann-Whitney U test
    mean_dcd = dcd.mean()
    mean_control = control.mean()
    std_dcd = dcd.std()
    std_control = control.std()
    resultsmw = mannwhitneyu(dcd, control)

    # Store the results in a dictionary
    results_mw1.append({
        'feature': name,
        'pvalue': resultsmw.pvalue,
        'stat': resultsmw.statistic,
        'n_dcd': n_dcd,
        'n_control': n_control,
        'mean_rank_dcd': mean_ranks[0],
        'mean_rank_control': mean_ranks[1],
        'mean_dcd': mean_dcd,
        'mean_control': mean_control,
        'std_dcd': std_dcd,
        'std_control': std_control
    })

# Convert the results to a DataFrame and save it to a CSV file
results_mw1_df = pd.DataFrame(results_mw1)
results_mw1_df.to_csv('./data/statistical/mw_results_dcd_control.csv', index=False)

# Perform Mann-Whitney U test: Ataxia vs Control
results_mw2 = []
for name, filepath in datasets.items():
    df = pd.read_csv(filepath)
    ataxia = df[df['label'] == 'EOA'][name]
    control = df[df['label'] == 'Control'][name]
    n_ataxia = len(ataxia)
    n_control = len(control)

    # Calculate the mean ranks for each group
    combined_ranks = pd.concat([ataxia.rank(), control.rank()])
    mean_ranks = [combined_ranks.loc[ataxia.index].mean(), combined_ranks.loc[control.index].mean()]

    # Calculate the mean, standard deviation, and perform the Mann-Whitney U test
    mean_ataxia = ataxia.mean()
    mean_control = control.mean()
    std_ataxia = ataxia.std()
    std_control = control.std()
    resultsmw = mannwhitneyu(ataxia, control)

    # Store the results in a dictionary
    results_mw2.append({
        'feature': name,
        'pvalue': resultsmw.pvalue,
        'stat': resultsmw.statistic,
        'n_ataxia': n_ataxia,
        'n_control': n_control,
        'mean_rank_ataxia': mean_ranks[0],
        'mean_rank_control': mean_ranks[1],
        'mean_ataxia': mean_ataxia,
        'mean_control': mean_control,
        'std_ataxia': std_ataxia,
        'std_control': std_control
    })

# Convert the results to a DataFrame and save it to a CSV file
results_mw2_df = pd.DataFrame(results_mw2)
results_mw2_df.to_csv('./data/statistical/mw_results_ataxia_control.csv', index=False)

# Perform Mann-Whitney U test: DCD vs Ataxia
results_mw3 = []
for name, filepath in datasets.items():
    df = pd.read_csv(filepath)
    ataxia = df[df['label'] == 'EOA'][name]
    dcd = df[df['label'] == 'DCD'][name]
    n_ataxia = len(ataxia)
    n_dcd = len(dcd)

    # Calculate the mean ranks for each group
    combined_ranks = pd.concat([ataxia.rank(), dcd.rank()])
    mean_ranks = [combined_ranks.loc[ataxia.index].mean(), combined_ranks.loc[dcd.index].mean()]

    # Calculate the mean, standard deviation, and perform the Mann-Whitney U test
    mean_ataxia = ataxia.mean()
    mean_dcd = dcd.mean()
    std_ataxia = ataxia.std()
    std_dcd = dcd.std()
    resultsmw = mannwhitneyu(ataxia, dcd)

    # Store the results in a dictionary
    results_mw3.append({
        'feature': name,
        'pvalue': resultsmw.pvalue,
        'stat': resultsmw.statistic,
        'n_ataxia': n_ataxia,
        'n_dcd': n_dcd,
        'mean_rank_ataxia': mean_ranks[0],
        'mean_rank_dcd': mean_ranks[1],
        'mean_ataxia': mean_ataxia,
        'mean_dcd': mean_dcd,
        'std_ataxia': std_ataxia,
        'std_dcd': std_dcd
    })

# Convert the results to a DataFrame and save it to a CSV file
results_mw3_df = pd.DataFrame(results_mw3)
results_mw3_df.to_csv('./data/statistical/mw_results_ataxia_dcd.csv', index=False)
