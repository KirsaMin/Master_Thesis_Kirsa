import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks, butter, sosfiltfilt

# Read the data file
data1 = pd.read_csv('/Users/kirsamin/PycharmProjects/test/data/ori_gait_normed_84.csv')

# Get the unique group values
group_list1 = data1.group.unique()

# Create directories for saving the graphs and data
os.makedirs('./data/graphs/distance_knee_ankle', exist_ok=True)
os.makedirs('./data/distance_knee_ankle', exist_ok=True)

# Process the left side data
timestamps_dict1 = {}
timestamps_dict2 = {}
for group_id in group_list1:
    # Filter the data and calculate the distance between left knee and ankle
    df = data1[data1.group == group_id].reset_index(drop=True)
    file_path = f'./data/graphs/distance_knee_ankle/LEFT_{group_id}.png'
    sos = butter(3, 1.75, 'lp', fs=30, output='sos')
    x16, y16 = df['LAnkle_x'], df['LAnkle_y']
    x17, y17 = df['LKnee_x'], df['LKnee_y']
    distance = np.sqrt((x17 - x16) ** 2 + (y17 - y16) ** 2)

    # Normalize the distance and apply a low-pass filter
    distance_normalized = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
    smoothl = sosfiltfilt(sos, distance_normalized)
    plt.plot(df.index, distance_normalized, color='#FFE569', label='not filtered')
    plt.plot(df.index, smoothl, color='#F79327', label='filtered')

    # Find peaks and troughs in the filtered data
    peaks1, _ = find_peaks(smoothl)
    peaks2, _ = find_peaks(-smoothl)

    # Calculate the thresholds for excluding outliers
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
    # plt.plot(df.index[peaks1], smoothl[peaks1], "x", color='#0079FF')
    # plt.plot(df.index[peaks2], smoothl[peaks2], "x", color='#00DFA2')

    # Set the graph title, labels, and legend
    plt.xlabel('frame')
    plt.ylabel('distance')
    plt.title(f"euclidean distance between left knee and ankle %s" % group_id)
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
timestamps1.to_csv('./data/distance_knee_ankle/LEFT_distance_knee_ankle_peaks_filt.csv')
timestamps2.to_csv('./data/distance_knee_ankle/LEFT_distance_knee_ankle_troughs_filt.csv')

# Process the right side data with normalized data
timestamps_dict1 = {}
timestamps_dict2 = {}
for group_id in group_list1:
    # Filter the data and calculate the distance between right knee and ankle
    df = data1[data1.group == group_id].reset_index(drop=True)
    file_path = f'./data/graphs/distance_knee_ankle/RIGHT_{group_id}.png'
    x16, y16 = df['RAnkle_x'], df['RAnkle_y']
    x17, y17 = df['RKnee_x'], df['RKnee_y']
    distance = np.sqrt((x17 - x16) ** 2 + (y17 - y16) ** 2)

    # Apply a low-pass filter to the distance data
    sos = butter(3, 1.75, 'lp', fs=30, output='sos')
    distance_normalized = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
    smoothl = sosfiltfilt(sos, distance_normalized)
    plt.plot(df.index, distance_normalized, color='#FFE569', label='not filtered')
    plt.plot(df.index, smoothl, color='#F79327', label='filtered')

    # Find peaks and troughs in the filtered data
    peaks1, _ = find_peaks(smoothl)
    peaks2, _ = find_peaks(-smoothl)

    # Calculate the thresholds for excluding outliers
    q1 = np.percentile(smoothl, 25)
    q3 = np.percentile(smoothl, 75)
    iqr = q3 - q1
    threshold2 = q1 - 1.5 * iqr
    threshold1 = q3 + 1.5 * iqr

    # Apply thresholds to exclude outliers and plot the filtered peaks and troughs
    filtered_peaks1 = peaks1[smoothl[peaks1] < threshold1]
    filtered_peaks2 = peaks2[smoothl[peaks2] > threshold2]
    plt.plot(df.index[filtered_peaks1], smoothl[filtered_peaks1], "x", color='#0079FF')
    plt.plot(df.index[filtered_peaks2], smoothl[filtered_peaks2], "x", color='#00DFA2')
    # plt.plot(df.index[peaks1], smoothl[peaks1], "x", color='#0079FF')
    # plt.plot(df.index[peaks2], smoothl[peaks2], "x", color='#00DFA2')

    # Set the graph title, labels, and legend
    plt.xlabel('frame')
    plt.ylabel('distance')
    plt.title(f"euclidean distance between right knee and ankle %s" % group_id)
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
timestamps1.to_csv('./data/distance_knee_ankle/RIGHT_distance_knee_ankle_peaks_filt.csv')
timestamps2.to_csv('./data/distance_knee_ankle/RIGHT_distance_knee_ankle_troughs_filt.csv')
