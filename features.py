import os
import pandas as pd

# Read the data from a CSV file
data = pd.read_csv('ground_truth_labels.csv')

# Create a directory to store the generated features
os.makedirs('./data/features', exist_ok=True)

# Get unique group values
group_list = data.group.unique()

# Create a dictionary to map group labels
label_dict = {}
for group in group_list:
    # Check if the group contains 'ataxia' label
    if (data[data['group'] == group]['label'] == 'ataxia').any():
        label_dict[group] = 'EOA'
    # Check if the group contains 'dcd' label
    elif (data[data['group'] == group]['label'] == 'dcd').any():
        label_dict[group] = 'DCD'
    # Check if the group contains 'controls' label
    elif (data[data['group'] == group]['label'] == 'controls').any():
        label_dict[group] = 'Control'
    else:
        label_dict[group] = 'unknown'

# Step time left --> left toe-off till right toe-off
step_time_left_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    step_times_left = []
    for i in range(5, 63, 4):
        step_time_left = (df.iloc[:,i] - df.iloc[:, i-2])
        for time in step_time_left:
            step_time_left_dict.setdefault(group_id, []).append(time)

# Prepare the rows for the step time left dataframe
rows = []
for group_id, step_times in step_time_left_dict.items():
    label = label_dict[group_id]
    for step_time in step_times:
        rows.append((group_id, label, step_time))
step_times_left = pd.DataFrame(rows, columns=['group_id', 'label', 'step_time_left'])

# Drop rows with NaN values and save the dataframe to a CSV file
step_times_left = step_times_left.dropna(subset=['step_time_left'])
step_times_left.to_csv('./data/features/step_time_left.csv', index=False)

# Step time right --> right toe-off till left toe-off
step_time_right_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    step_times_right = []
    for i in range(7, 63, 4):
        step_time_right = (df.iloc[:,i] - df.iloc[:, i-2])
        for time in step_time_right:
            step_time_right_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, step_times in step_time_right_dict.items():
    label = label_dict[group_id]
    for step_time in step_times:
        rows.append((group_id, label, step_time))
step_times_right = pd.DataFrame(rows, columns=['group_id', 'label', 'step_time_right'])
step_times_right = step_times_right.dropna(subset=['step_time_right'])
step_times_right.to_csv('./data/features/step_time_right.csv', index=False)

# Stance time left --> left heel-strike till left toe-off
stance_time_left_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    stance_times_left = []
    for i in range(7, 63, 4):
        stance_time_left = (df.iloc[:,i] - df.iloc[:, i-3])
        for time in stance_time_left:
            stance_time_left_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, stance_times in stance_time_left_dict.items():
    label = label_dict[group_id]
    for stance_time in stance_times:
        rows.append((group_id, label, stance_time))
stance_times_left = pd.DataFrame(rows, columns=['group_id', 'label', 'stance_time_left'])
stance_times_left = stance_times_left.dropna(subset=['stance_time_left'])
stance_times_left.to_csv('./data/features/stance_time_left.csv', index=False)

# Stance time right --> right heel-strike till right toe-off
stance_time_right_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    stance_times_right = []
    for i in range(5, 63, 4):
        stance_time_right = (df.iloc[:,i] - df.iloc[:, i-3])
        for time in stance_time_right:
            stance_time_right_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, stance_times in stance_time_right_dict.items():
    label = label_dict[group_id]
    for stance_time in stance_times:
        rows.append((group_id, label, stance_time))
stance_times_right = pd.DataFrame(rows, columns=['group_id', 'label', 'stance_time_right'])
stance_times_right = stance_times_right.dropna(subset=['stance_time_right'])
stance_times_right.to_csv('./data/features/tance_time_right.csv', index=False)

# Swing time left --> left toe-off till left heel-strike
swing_time_left_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    swing_times_left = []
    for i in range(4, 63, 4):
        swing_time_left = (df.iloc[:,i] - df.iloc[:, i-1])
        for time in swing_time_left:
            swing_time_left_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, swing_times in swing_time_left_dict.items():
    label = label_dict[group_id]
    for swing_time in swing_times:
        rows.append((group_id, label, swing_time))
swing_times_left = pd.DataFrame(rows, columns=['group_id', 'label', 'swing_time_left'])
swing_times_left = swing_times_left.dropna(subset=['swing_time_left'])
swing_times_left.to_csv('./data/features/swing_time_left.csv', index=False)

# Swing time right --> right toe-off till right heel-strike
swing_time_right_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    swing_times_right = []
    for i in range(6, 63, 4):
        swing_time_right = (df.iloc[:,i] - df.iloc[:, i-1])
        for time in swing_time_right:
            swing_time_right_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, swing_times in swing_time_right_dict.items():
    label = label_dict[group_id]
    for swing_time in swing_times:
        rows.append((group_id, label, swing_time))
swing_times_right = pd.DataFrame(rows, columns=['group_id', 'label', 'swing_time_right'])
swing_times_right = swing_times_right.dropna(subset=['swing_time_right'])
swing_times_right.to_csv('./data/features/swing_time_right.csv', index=False)

# Stride time left --> left heel-strike till left heel-strike
stride_time_left_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    stride_times_left = []
    for i in range(8, 63, 4):
        stride_time_left = (df.iloc[:,i] - df.iloc[:, i-4])
        for time in stride_time_left:
            stride_time_left_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, stride_times in stride_time_left_dict.items():
    label = label_dict[group_id]
    for stride_time in stride_times:
        rows.append((group_id, label, stride_time))
stride_times_left = pd.DataFrame(rows, columns=['group_id', 'label', 'stride_time_left'])
stride_times_left = stride_times_left.dropna(subset=['stride_time_left'])
stride_times_left.to_csv('./data/features/stride_time_left.csv', index=False)

# Stride time right --> right heel-strike till right heel-strike
stride_time_right_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    stride_times_right = []
    for i in range(6, 63, 4):
        stride_time_right = (df.iloc[:,i] - df.iloc[:, i-4])
        for time in stride_time_right:
            stride_time_right_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, stride_times in stride_time_right_dict.items():
    label = label_dict[group_id]
    for stride_time in stride_times:
        rows.append((group_id, label, stride_time))
stride_times_right = pd.DataFrame(rows, columns=['group_id', 'label', 'stride_time_right'])
stride_times_right = stride_times_right.dropna(subset=['stride_time_right'])
stride_times_right.to_csv('./data/features/stride_time_right.csv', index=False)

# Double support --> right heel-strike till left toe-off and left heel-strike till right toe-off
double_support_time_dict = {}
for group_id in group_list:
    df = data[data['group'] == group_id]
    double_support_times = []
    for i in range(3, 63, 2):
        double_support_time = (df.iloc[:,i] - df.iloc[:, i-1])
        for time in double_support_time:
            double_support_time_dict.setdefault(group_id, []).append(time)

rows = []
for group_id, double_support_times in double_support_time_dict.items():
    label = label_dict[group_id]
    for double_support_time in double_support_times:
        rows.append((group_id, label, double_support_time))
double_support_times = pd.DataFrame(rows, columns=['group_id', 'label', 'double_support_time'])
double_support_times = double_support_times.dropna(subset=['double_support_time'])
double_support_times.to_csv('./data/features/double_support_time.csv', index=False)
