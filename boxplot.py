import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from pylab import *
import numpy as np

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

# Create a directory for saving the graphs
os.makedirs('./data/graphs/features', exist_ok=True)

# Iterate over each dataset
for name, filepath in datasets.items():
    # Define the output path for the graph
    path = f'./data/graphs/features/{name}.png'

    # Read the dataset
    df = pd.read_csv(filepath)

    # Group the data by label and extract the feature values as lists
    grouped_data = df.groupby('label')[name].apply(list).reset_index()

    # Create a figure and axis for plotting
    fig = figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Create a boxplot of the feature values for each label
    bp = ax.boxplot(grouped_data[name])

    # Set the title, y-axis label, and x-axis tick labels
    plt.title(f"%s" % name)
    plt.ylabel('time (s)')
    plt.xticks(range(1, len(grouped_data) + 1), grouped_data['label'])

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove the left spine and set grid lines
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)

    # Set the y-axis limits
    ylim = np.max(np.concatenate((df[df['label'] == 'EOA'][name], df[df['label'] == 'Control'][name],
                                  df[df['label'] == 'DCD'][name]))) * 1.5
    ax.set_ylim([0, ylim])

    # Set the grid lines below the boxplot
    ax.set_axisbelow(True)

    # Perform Kruskal-Wallis test to compare all three groups
    k, p = stats.kruskal(df[df['label'] == 'EOA'][name],
                         df[df['label'] == 'Control'][name],
                         df[df['label'] == 'DCD'][name])

    # Add a text annotation with the Kruskal-Wallis p-value
    ax.text(0.05, 0.9, f'kruskal-wallis p={p:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Perform Mann-Whitney U tests to compare different groups
    z1, p1 = stats.mannwhitneyu(df[df['label'] == 'EOA'][name],
                                df[df['label'] == 'Control'][name])
    z2, p2 = stats.mannwhitneyu(df[df['label'] == 'DCD'][name],
                                df[df['label'] == 'Control'][name])
    z3, p3 = stats.mannwhitneyu(df[df['label'] == 'EOA'][name],
                                df[df['label'] == 'DCD'][name])

    # Calculate the maximum and minimum values for each group
    y_max_ac = np.max(np.concatenate((df[df['label'] == 'EOA'][name],
                                      df[df['label'] == 'Control'][name])))
    y_min_ac = np.min(np.concatenate((df[df['label'] == 'EOA'][name],
                                      df[df['label'] == 'Control'][name])))
    y_max_dc = np.max(np.concatenate((df[df['label'] == 'DCD'][name],
                                      df[df['label'] == 'Control'][name])))
    y_min_dc = np.min(np.concatenate((df[df['label'] == 'DCD'][name],
                                      df[df['label'] == 'Control'][name])))
    y_max_ad = np.max(np.concatenate((df[df['label'] == 'EOA'][name],
                                      df[df['label'] == 'DCD'][name])))
    y_min_ad = np.min(np.concatenate((df[df['label'] == 'EOA'][name],
                                      df[df['label'] == 'DCD'][name])))

    # Add arrows and text annotations for Mann-Whitney U test results
    ax.annotate("", xy=(1, y_max_dc), xycoords='data',
                xytext=(2, y_max_dc), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                connectionstyle="bar,fraction=0.05"))
    ax.text(1.5, y_max_dc * 1.05, f'p={p2:.3f}',
            horizontalalignment='center', verticalalignment='bottom')
    ax.annotate("", xy=(2, y_max_ad), xycoords='data',
                xytext=(3, y_max_ad), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                connectionstyle="bar,fraction=0.1"))
    ax.text(2.5, y_max_ad * 1.05, f'p={p3:.3f}',
            horizontalalignment='center', verticalalignment='bottom')
    ax.annotate("", xy=(1, y_max_ac * 1.1), xycoords='data',
                xytext=(3, y_max_ac * 1.1), textcoords='data',
                arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                connectionstyle="bar,fraction=0.05"))
    ax.text(2, y_max_ac * 1.15, f'p={p1:.3f}',
            horizontalalignment='center', verticalalignment='bottom')

    # Save the figure as an image
    plt.savefig(path)
    plt.clf()
