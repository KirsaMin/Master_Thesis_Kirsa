import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks, butter, sosfiltfilt

# Read the data from the CSV file
data1 = pd.read_csv('/Users/kirsamin/PycharmProjects/test/data/ori_gait_normed_84.csv')

# Get unique group IDs
group_list1 = data1.group.unique()

# Create directories to store graphs and data
os.makedirs('./data/graphs/y_coordinates/pair', exist_ok=True)
os.makedirs('./data/y_coordinates', exist_ok=True)

# Dictionary to store timestamps of peaks and troughs
timestamps_dict1 = {}
timestamps_dict2 = {}

# Process left side data
for group_id in group_list1:
    # Get data for the current group
    df = data1[data1.group == group_id].reset_index(drop=True)
    file_path = f'./data/graphs/y_coordinates/LEFT_{group_id}.png'

    # Apply a low-pass filter to normalize the y-coordinate data
    sos = butter(3, 1.75, 'lp', fs=30, output='sos')
    y_normalized = (df['LAnkle_y'] - np.min(df['LAnkle_y'])) / (np.max(df['LAnkle_y']) - np.min(df['LAnkle_y']))
    smoothl = sosfiltfilt(sos, y_normalized)

    # Plot the original and filtered data
    plt.plot(df.index, y_normalized, color='#FFE569', label='not filtered')
    plt.plot(df.index, smoothl, color='#F79327', label='filtered')

    # Find peaks and troughs in the filtered data
    peaks1, _ = find_peaks(smoothl)
    peaks2, _ = find_peaks(-smoothl)

    # Calculate thresholds to exclude outliers
    q1 = np.percentile(smoothl, 25)
    q3 = np.percentile(smoothl, 75)
    iqr = q3 - q1
    threshold1 = q1 - 1.5 * iqr
    threshold2 = q3 + 1.5 * iqr

    # Apply thresholds to exclude outliers and plot the filtered peaks and troughs
    filtered_peaks1 = peaks1[smoothl[peaks1] > threshold1]
    filtered_peaks2 = peaks2[smoothl[peaks2] < threshold2]
    plt.plot(df.index[filtered_peaks1], smoothl[filtered_peaks1], "x", color='#0079FF')
    plt.plot(df.index[filtered_peaks2], smoothl[filtered_peaks2], "x", color='#00DFA2')
    # plt.plot(df.index[peaks1], smoothl[peaks1], "x", color='#00DFA2')
    # plt.plot(df.index[peaks2], smoothl[peaks2], "x", color='#0079FF')

    # Invert the y-axis for visualization
    plt.gca().invert_yaxis()

    # Set the graph title, labels, and legend
    plt.xlabel('frame')
    plt.ylabel('y-coordinate')
    plt.title(f"y-coordinates left ankle %s" % group_id)
    plt.legend()

    # Save the graph as an image
    plt.savefig(file_path)
    plt.clf()

    # Calculate the timestamps for the filtered peaks and troughs
    time1 = df.index[filtered_peaks1] / 30
    time2 = (df.index[filtered_peaks2] / 30) - 0.1775
    # time1 = df.index[peaks1] / 30
    # time2 = (df.index[peaks2] / 30) - 0.1775

    # Store the timestamps in a dictionary
    timestamps_dict1[group_id] = time1
    timestamps_dict2[group_id] = time2

# Create data frames from the timestamps dictionaries
timestamps1 = pd.DataFrame.from_dict(timestamps_dict1, orient='index').transpose()
timestamps2 = pd.DataFrame.from_dict(timestamps_dict2, orient='index').transpose()

# Save the timestamps data frames as CSV files
timestamps1.to_csv('./data/y_coordinates/LEFT_y_coordinates_troughs_filt.csv')
timestamps2.to_csv('./data/y_coordinates/LEFT_y_coordinates_peaks_filt.csv')

# Process right side data
timestamps_dict1 = {}
timestamps_dict2 = {}
for group_id in group_list1:
    # Get data for the current group
    df = data1[data1.group == group_id].reset_index(drop=True)
    file_path = f'./data/graphs/y_coordinates/RIGHT_{group_id}.png'

    # Apply a low-pass filter to normalize the y-coordinate data
    sos = butter(3, 1.75, 'lp', fs=30, output='sos')
    y_normalized = (df['RAnkle_y'] - np.min(df['RAnkle_y'])) / (np.max(df['RAnkle_y']) - np.min(df['RAnkle_y']))
    smoothr = sosfiltfilt(sos, y_normalized)

    # Plot the original and filtered data
    plt.plot(df.index, y_normalized, color='#FFE569', label='not filtered')
    plt.plot(df.index, smoothr, color='#F79327', label='filtered')

    # Find peaks and troughs in the filtered data
    peaks1, _ = find_peaks(smoothr)
    peaks2, _ = find_peaks(-smoothr)

    # Calculate thresholds to exclude outliers
    q1 = np.percentile(smoothr, 25)
    q3 = np.percentile(smoothr, 75)
    iqr = q3 - q1
    threshold1 = q1 - 1.5 * iqr
    threshold2 = q3 + 1.5 * iqr

    # Apply thresholds to exclude outliers and plot the filtered peaks and troughs
    filtered_peaks1 = peaks1[smoothr[peaks1] > threshold1]
    filtered_peaks2 = peaks2[smoothr[peaks2] < threshold2]
    plt.plot(df.index[filtered_peaks1], smoothr[filtered_peaks1], "x", color='#0079FF')
    plt.plot(df.index[filtered_peaks2], smoothr[filtered_peaks2], "x", color='#00DFA2')
    # plt.plot(df.index[peaks1], smoothr[peaks1], "x", color='#00DFA2')
    # plt.plot(df.index[peaks2], smoothr[peaks2], "x", color='#0079FF')

    # Invert the y-axis for visualization
    plt.gca().invert_yaxis()

    # Set the graph title, labels, and legend
    plt.xlabel('frame')
    plt.ylabel('y-coordinate')
    plt.title(f"y-coordinates right ankle %s" % group_id)
    plt.legend()

    # Save the graph as an image
    plt.savefig(file_path)
    plt.clf()

    # Calculate the timestamps for the filtered peaks and troughs
    time1 = df.index[filtered_peaks1] / 30
    time2 = (df.index[filtered_peaks2] / 30) - 0.1775
    # time1 = df.index[peaks1] / 30
    # time2 = (df.index[peaks2] / 30) - 0.1775

    # Store the timestamps in a dictionary
    timestamps_dict1[group_id] = time1
    timestamps_dict2[group_id] = time2

# Create data frames from the timestamps dictionaries
timestamps1 = pd.DataFrame.from_dict(timestamps_dict1, orient='index').transpose()
timestamps2 = pd.DataFrame.from_dict(timestamps_dict2, orient='index').transpose()

# Save the timestamps data frames as CSV files
timestamps1.to_csv('./data/y_coordinates/RIGHT_y_coordinates_troughs_filt.csv')
timestamps2.to_csv('./data/y_coordinates/RIGHT_y_coordinates_peaks_filt.csv')
