import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import sklearn.metrics


# Step 1: Data Loading
# Load CSV file
data_df = pd.read_csv('/Users/kirsamin/PycharmProjects/test/data/ori_gait_normed_84.csv')


# Normalize data between 0-1
scaler = MinMaxScaler()
excluded_columns = ['group', 'label', 'patient']
scaler = MinMaxScaler()
columns_to_normalize = [col for col in data_df.columns if col not in excluded_columns]
data_df[columns_to_normalize] = scaler.fit_transform(data_df[columns_to_normalize])


# Ground truth labels (left-toe-off, left-heel-strike, right-toe-off, right-heel-strike)
labels_df = pd.read_csv('/Users/kirsamin/PycharmProjects/test/data/labels_gt.csv')


# Forward-fill NaN values in the labels dataframe
labels_df = labels_df.ffill(axis=1)


# Extract the group information from the labels dataframe
groups = np.unique(data_df.group.values)


# Convert the time series labels to frame-level labels
fps = 30  # Frames per second


# Calculate the total frames across all groups
total_frames = sum(len(data_df[data_df['group'] == group]) for group in groups)


# Determine the fixed length for evaluation and training
fixed_length = min(len(data_df[data_df['group'] == group]) for group in groups)-1


def generate_consecutive_sample(data, group, fixed_length=50):
   # Generate consecutive indices for the given group
   group_data = data[data['group'] == group].reset_index(drop=True)
   max_start_index = len(group_data) - fixed_length
   start_index = np.random.choice(max_start_index)
   indices_group = np.arange(start_index, start_index + fixed_length)
   group_data = group_data[np.isin(group_data.index, indices_group)].iloc[:,-9:-3]
   group_labels = labels_df.loc[labels_df['group'] == group].iloc[:, 1:]
   group_labels = np.round(group_labels * fps).astype(int)
   group_labels = group_labels[group_labels.isin(indices_group)].dropna(axis=1)


   # Build a (fixed_length, 4) array for rhs, lto, lhs, rto
   labels_array = np.zeros((fixed_length, 4))
   for i, label in enumerate(group_labels):
       start_step = int(group_labels.columns[0][-1])
       if not group_labels.empty:
           frame_index = group_labels[label].values[0]-start_index
           if label[:3]== 'rhs':
               labels_array[frame_index, 0] = 1
           elif label[:3]== 'lto':
               labels_array[frame_index, 1] = 1
           elif label[:3]== 'lhs':
               labels_array[frame_index, 2] = 1
           elif label[:3]== 'rto':
               labels_array[frame_index, 3] = 1
   return np.array(group_data), labels_array


# Get unique groups
unique_groups = np.unique(groups)

# Randomly split the groups into training and evaluation groups
np.random.shuffle(unique_groups)
train_groups = unique_groups[:int(0.8 * len(unique_groups))]  # 80% for training
eval_groups = unique_groups[int(0.8 * len(unique_groups)):]  # 20% for evaluation
resampling_times = 4

def obtain_resampled_data(data_df,groups):
   data, labels = [], []
   for group in groups:
       for i in range(resampling_times):
           group_data, group_labels = generate_consecutive_sample(data_df, group, fixed_length)
           data.append(group_data)
           labels.append(group_labels)
   return np.array(data), np.array(labels)

training_data, training_labels = obtain_resampled_data(data_df,train_groups)
eval_data, eval_labels = obtain_resampled_data(data_df,eval_groups)

# Example of training and evaluation data and labels
print("Training Data Shape:", training_data.shape)
print("Training Labels Shape:", training_labels.shape)
print("Evaluation Data Shape:", eval_data.shape)
print("Evaluation Labels Shape:", eval_labels.shape)

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
   def __init__(self, data, labels):
       self.data = data
       self.labels = labels

   def __len__(self):
       return len(self.data)

   def __getitem__(self, idx):
       x = self.data[idx]
       y = self.labels[idx]
       return x, y

train_dataset = CustomDataset(training_data, training_labels)
eval_dataset = CustomDataset(eval_data, eval_labels)

class LSTMModel(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
       super(LSTMModel, self).__init__()
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
       self.dropout = nn.Dropout(dropout_rate)
       self.fc = nn.Linear(hidden_size * 2, output_size)

   def forward(self, x):
       batch_size = x.size(0)
       seq_length = x.size(1)
       h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
       c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
       x = x.to(dtype=torch.float32)  # Convert input to float32
       out, _ = self.lstm(x, (h0, c0))
       out = self.dropout(out)
       out = self.fc(out)
       return out

input_size = training_data.shape[2]
hidden_size = 512
num_layers = 2
output_size = training_labels.shape[2]
learning_rate = 0.001
num_epochs = 100
dropout_rate = 0.3

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize lists to store metrics
train_loss_list = []
eval_loss_list = []
train_accuracy_list = []
eval_accuracy_list = []
train_precision_list = []
eval_precision_list = []
train_recall_list = []
eval_recall_list = []
train_f1_list = []
eval_f1_list = []
train_precision_curve = []
eval_precision_curve = []
train_recall_curve = []
eval_recall_curve = []

for epoch in range(num_epochs):
   model.train()
   total_train_correct = 0
   total_train_samples = 0
   train_tp = 0
   train_tn = 0
   train_fp = 0
   train_fn = 0

   for batch_data, batch_labels in train_loader:
       batch_data = batch_data.to(device).float()
       batch_labels = batch_labels.to(device).float()

       optimizer.zero_grad()

       outputs = model(batch_data)
       loss = criterion(outputs, batch_labels)

       loss.backward()
       optimizer.step()

       # Calculate training accuracy
       train_predictions = torch.round(outputs)

       # Calculate training precision and recall
       train_tp += ((train_predictions == 1) & (batch_labels == 1)).sum().item()
       train_tn += ((train_predictions == 0) & (batch_labels == 0)).sum().item()
       train_fp += ((train_predictions == 1) & (batch_labels == 0)).sum().item()
       train_fn += ((train_predictions == 0) & (batch_labels == 1)).sum().item()


   train_accuracy = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn) if (train_tp + train_tn + train_fp + train_fn) \
                                                                                           != 0 else 0.0
   train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) != 0 else 0.0
   train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) != 0 else 0.0
   train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) \
                                                                                         != 0 else 0.0

   model.eval()
   eval_loss = 0.0
   total_eval_correct = 0
   total_eval_samples = 0
   eval_tp = 0
   eval_tn = 0
   eval_fp = 0
   eval_fn = 0

   with torch.no_grad():
       for batch_data, batch_labels in eval_loader:
           batch_data = batch_data.to(device).float()
           batch_labels = batch_labels.to(device).float()


           outputs = model(batch_data)
           eval_loss += criterion(outputs, batch_labels).item()


           # Calculate evaluation accuracy
           eval_predictions = torch.round(outputs)

           # Calculate evaluation precision and recall
           eval_tp += ((eval_predictions == 1) & (batch_labels == 1)).sum().item()
           eval_tn += ((eval_predictions == 0) & (batch_labels == 0)).sum().item()
           eval_fp += ((eval_predictions == 1) & (batch_labels == 0)).sum().item()
           eval_fn += ((eval_predictions == 0) & (batch_labels == 1)).sum().item()


   eval_loss /= len(eval_loader)
   eval_accuracy = (eval_tp + eval_tn) / (eval_tp + eval_tn + eval_fp + eval_fn) if (eval_tp + eval_tn + eval_fp + eval_fn) \
                                                                                    != 0 else 0.0
   eval_precision = eval_tp / (eval_tp + eval_fp) if (eval_tp + eval_fp) != 0 else 0.0
   eval_recall = eval_tp / (eval_tp + eval_fn) if (eval_tp + eval_fn) != 0 else 0.0
   eval_f1 = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall) if (eval_precision + eval_recall) \
                                                                                    != 0 else 0.0

   train_loss_list.append(loss.item())
   eval_loss_list.append(eval_loss)
   train_accuracy_list.append(train_accuracy)
   eval_accuracy_list.append(eval_accuracy)
   train_precision_list.append(train_precision)
   eval_precision_list.append(eval_precision)
   train_recall_list.append(train_recall)
   eval_recall_list.append(eval_recall)
   train_f1_list.append(train_f1)
   eval_f1_list.append(eval_f1)
   train_precision_curve.append(train_precision)
   eval_precision_curve.append(eval_precision)
   train_recall_curve.append(train_recall)
   eval_recall_curve.append(eval_recall)

   print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Eval Loss: {eval_loss:.4f},"
         f" Train Accuracy: {train_accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f},"
         f" Train Precision: {train_precision:.4f}, Eval Precision: {eval_precision:.4f},"
         f" Train Recall: {train_recall:.4f}, Eval Recall: {eval_recall:.4f},"
         f" Train F1: {train_f1:.4f}, Eval F1: {eval_f1:.4f}")

train_loss_mean = sum(train_loss_list) / len(train_loss_list)
eval_loss_mean = sum(eval_loss_list) / len(eval_loss_list)
train_accuracy_mean = sum(train_accuracy_list) / len(train_accuracy_list)
eval_accuracy_mean = sum(eval_accuracy_list) / len(eval_accuracy_list)
train_precision_mean = sum(train_precision_list) / len(train_precision_list)
eval_precision_mean = sum(eval_precision_list) / len(eval_precision_list)
train_recall_mean = sum(train_recall_list) / len(train_recall_list)
eval_recall_mean = sum(eval_recall_list) / len(eval_recall_list)
train_f1_mean = sum(train_f1_list) / len(train_f1_list)
eval_f1_mean = sum(eval_f1_list) / len(eval_f1_list)

print(f"Mean train loss: {train_loss_mean:.3f}, Mean eval loss: {eval_loss_mean:.3f},"
      f"Mean train accuracy: {train_accuracy_mean:.3f}, Mean eval accuracy: {eval_accuracy_mean:.3f},"
      f"Mean train precision: {train_precision_mean:.3f}, Mean eval precision: {eval_precision_mean:.3f},"
      f"Mean train recall: {train_recall_mean:.3f}, Mean eval recall: {eval_recall_mean:.3f},"
      f"Mean train F1-score: {train_f1_mean:.3f}, Mean eval F1-score: {eval_f1_mean:.3f}")

# Plot 1: Train loss and eval loss of each epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, num_epochs + 1), eval_loss_list, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss and Eval Loss')
plt.legend()
plt.show()

# Plot 2: Train accuracy and eval accuracy of each epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), eval_accuracy_list, label='Eval Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy and Eval Accuracy')
plt.legend()
plt.show()

# Plot 3: Train precision and eval precision of each epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), train_precision_list, label='Train Precision')
plt.plot(range(1, num_epochs + 1), eval_precision_list, label='Eval Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Train Precision and Eval Precision')
plt.legend()
plt.show()

# Plot 4: Train recall and eval recall of each epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), train_recall_list, label='Train Recall')
plt.plot(range(1, num_epochs + 1), eval_recall_list, label='Eval Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Train Recall and Eval Recall')
plt.legend()
plt.show()

# Plot 5: Train F1 score and eval F1 score of each epoch
plt.figure()
plt.plot(range(1, num_epochs + 1), train_f1_list, label='Train F1 Score')
plt.plot(range(1, num_epochs + 1), eval_f1_list, label='Eval F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Train F1 Score and Eval F1 Score')
plt.legend()
plt.show()