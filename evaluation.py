import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Define the file paths for dataset1
dataset1 = {
    # 'LEFT_y_coordinates_peaks': 'data/y_coordinates/LEFT_y_coordinates_peaks.csv',
    # 'LEFT_y_coordinates_troughs': 'data/y_coordinates/LEFT_y_coordinates_troughs.csv',
    # 'LEFT_distance_knee_ankle_peaks': 'data/distance_knee_ankle/LEFT_distance_knee_ankle_peaks.csv'
    # 'LEFT_distance_knee_ankle_troughs': 'data/distance_knee_ankle/LEFT_distance_knee_ankle_troughs.csv'
    # 'left_y_coordinates_peaks_filt': 'data/y_coordinates/LEFT_y_coordinates_peaks_filt.csv',
    'left_y_coordinates_troughs_filt': 'data/y_coordinates/LEFT_y_coordinates_troughs_filt.csv',
    'left_distance_knee_ankle_peaks_filt': 'data/distance_knee_ankle/LEFT_distance_knee_ankle_peaks_filt.csv'
    # 'left_distance_knee_ankle_troughs_filt': 'data/distance_knee_ankle/LEFT_distance_knee_ankle_troughs_filt.csv'
}

# Define the file paths for dataset2
dataset2 = {
    # 'RIGHT_y_coordinates_peaks': 'data/y_coordinates/RIGHT_y_coordinates_peaks.csv',
    # 'RIGHT_y_coordinates_troughs': 'data/y_coordinates/RIGHT_y_coordinates_troughs.csv',
    # 'RIGHT_distance_knee_ankle_peaks': 'data/distance_knee_ankle/RIGHT_distance_knee_ankle_peaks.csv'
    # 'RIGHT_distance_knee_ankle_troughs': 'data/distance_knee_ankle/RIGHT_distance_knee_ankle_troughs.csv'
    # 'right_y_coordinates_peaks_filt':'data/y_coordinates/RIGHT_y_coordinates_peaks_filt.csv',
    'right_y_coordinates_troughs_filt': 'data/y_coordinates/RIGHT_y_coordinates_troughs_filt.csv',
    'right_distance_knee_ankle_peaks_filt': 'data/distance_knee_ankle/RIGHT_distance_knee_ankle_peaks_filt.csv'
    # 'right_distance_knee_ankle_troughs_filt': 'data/distance_knee_ankle/RIGHT_distance_knee_ankle_troughs_filt.csv'
}

# Define the file paths for dataset3
dataset3 = {
    # 'lto_22': 'data/lto_22_transposed.csv'
    'lhs_22': 'data/lhs_22_transposed.csv'
}

# Define the file paths for dataset4
dataset4 = {
    # 'rto_22': 'data/rto_22_transposed.csv'
    'rhs_22': 'data/rhs_22_transposed.csv'
}

# Create directories to store the output files
os.makedirs('./data/graphs/compare/diff_index', exist_ok=True)
os.makedirs('./data/compare', exist_ok=True)

# Comparing dataset3 with dataset1
for file3_name, file3_path in dataset3.items():
    data3 = pd.read_csv(file3_path)

    for file1_name, file1_path in dataset1.items():
        data1 = pd.read_csv(file1_path)
        common_cols = list(set(data3.columns).intersection(set(data1.columns)))

        for col in common_cols:
            # Plot the data from dataset3 and dataset1
            plt.scatter(data3.index, data3[col], color='#FDD36A', label=file3_name)
            plt.scatter(data1.index, data1[col], color='#DC8449', label=file1_name)
            plt.ylabel('timestamp (s)')
            plt.title('{} Comparison ({} vs. {})'.format(col, file3_name, file1_name))
            plt.legend()
            plt.savefig('data/graphs/compare/{}_{}_{}_comparison.png'.format(file3_name, file1_name, col))
            plt.clf()

# Comparing dataset4 with dataset2
for file4_name, file4_path in dataset4.items():
    data4 = pd.read_csv(file4_path)

    for file2_name, file2_path in dataset2.items():
        data2 = pd.read_csv(file2_path)
        common_cols = list(set(data4.columns).intersection(set(data2.columns)))

        for col in common_cols:
            # Plot the data from dataset4 and dataset2
            plt.scatter(data4.index, data4[col], color='#FDD36A', label=file4_name)
            plt.scatter(data2.index, data2[col], color='#DC8449', label=file2_name)
            plt.ylabel('timestamp (s)')
            plt.title('{} Comparison ({} vs. {})'.format(col, file4_name, file2_name))
            plt.legend()
            plt.savefig('data/graphs/compare/{}_{}_{}_comparison.png'.format(file4_name, file2_name, col))
            plt.clf()

# Comparing dataset3 with dataset1 and calculating closest indices and errors
for file3_name, file3_path in dataset3.items():
    data3 = pd.read_csv(file3_path)
    data3 = data3.iloc[:, 1:]

    for file1_name, file1_path in dataset1.items():
        data1 = pd.read_csv(file1_path)
        data1 = data1.iloc[:, 1:]
        common_cols = list(set(data3.columns).intersection(set(data1.columns)))

        closest_indices = {}
        closest_errors = {}
        for col in common_cols:
            closest_indices[col] = []
            closest_errors[col] = []

            for index3, row3 in data3.iterrows():
                value3 = row3[col]
                min_diff = None
                closest_index = None
                closest_error = None

                for index1, row1 in data1.iterrows():
                    value1 = row1[col]
                    diff = abs(value3 - value1)

                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_index = index1
                        closest_error = diff

                closest_indices[col].append(closest_index)
                closest_errors[col].append(closest_error)

            # Plot the data from dataset1 and corresponding closest indices from dataset3
            plt.scatter(data1.index, data1[col], color='#DC8449', label=file1_name)
            plt.scatter(closest_indices[col], data3[col], color='#FDD36A', label=file3_name)
            plt.ylabel('timestamp (s)')
            plt.title('{} Comparison ({} vs. {})'.format(col, file3_name, file1_name))
            plt.legend()
            plt.savefig('data/graphs/compare/diff_index/{}_{}_{}_comparison.png'.format(file3_name, file1_name, col))
            plt.clf()

            errors_df = pd.DataFrame(closest_errors, columns=common_cols)
            errors_df.to_csv('./data/compare/{}_{}_errors.csv'.format(file3_name, file1_name), index=False)

# Comparing dataset4 with dataset2 and calculating closest indices and errors
for file4_name, file4_path in dataset4.items():
    data4 = pd.read_csv(file4_path)
    data4 = data4.iloc[:, 1:]

    for file2_name, file2_path in dataset2.items():
        data2 = pd.read_csv(file2_path)
        data2 = data2.iloc[:, 1:]
        common_cols = list(set(data4.columns).intersection(set(data2.columns)))

        closest_indices = {}
        closest_errors = {}
        for col in common_cols:
            closest_indices[col] = []
            closest_errors[col] = []

            for index4, row4 in data4.iterrows():
                value4 = row4[col]
                min_diff = None
                closest_index = None
                closest_error = None

                for index2, row2 in data2.iterrows():
                    value2 = row2[col]
                    diff = abs(value4 - value2)

                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_index = index2
                        closest_error = diff

                closest_indices[col].append(closest_index)
                closest_errors[col].append(closest_error)

            # Plot the data from dataset2 and corresponding closest indices from dataset4
            plt.scatter(data2.index, data2[col], color='#DC8449', label=file2_name)
            plt.scatter(closest_indices[col], data4[col], color='#FDD36A', label=file4_name)
            plt.ylabel('timestamp (s)')
            plt.title('{} Comparison ({} vs. {})'.format(col, file4_name, file2_name))
            plt.legend()
            plt.savefig('data/graphs/compare/diff_index/{}_{}_{}_comparison.png'.format(file4_name, file2_name, col))
            plt.clf()

            errors_df = pd.DataFrame(closest_errors, columns=common_cols)
            errors_df.to_csv('./data/compare/{}_{}_errors.csv'.format(file4_name, file2_name), index=False)

# Analyzing and ranking the mean errors
path = './data/compare'
csv_files = glob.glob(os.path.join(path, "*.csv"))

data = []
for f in csv_files:
    df = pd.read_csv(f)
    mean_rows = df.mean(axis=1)
    mean_csv = mean_rows.mean()
    data.append((f.split('/')[-1], mean_csv))

data.sort(key=lambda x: x[1])
for i, d in enumerate(data):
    print(f"Rank {i+1}: {d[0]}, mean: {d[1]}")
