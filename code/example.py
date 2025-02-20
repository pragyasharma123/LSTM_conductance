import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import generate_dataset
import lstm_encoder_decoder
import plotting
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load and preprocess conductance data
file_path = "/home/pshar188/LSTM_encoder_decoder/combined_data.csv"
data = pd.read_csv(file_path)

# Use only the measured conductance for modeling
conductance_data = data["measured_conductance"].values

# Apply smoothing to conductance data
from scipy.signal import savgol_filter
conductance_data_smoothed = savgol_filter(conductance_data, window_length=11, polyorder=2)

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
conductance_data_scaled = scaler.fit_transform(conductance_data_smoothed.reshape(-1, 1))

# Split into train/test sets
t = np.arange(len(conductance_data_scaled))  # Time indices
t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, conductance_data_scaled, split=0.8)

# Generate windowed dataset
input_window = 120
output_window = 20
stride = 2
Xtrain, Ytrain = generate_dataset.windowed_dataset(y_train, input_window=input_window, output_window=output_window, stride=stride)
Xtest, Ytest = generate_dataset.windowed_dataset(y_test, input_window=input_window, output_window=output_window, stride=stride)

# Convert to PyTorch tensors
X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# Train the model
model = lstm_encoder_decoder.lstm_seq2seq(input_size=X_train.shape[2], hidden_size=32).to(device)
'''
loss = model.train_model(
    X_train, Y_train, n_epochs=10, target_len=output_window, batch_size=32,
    training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.6, learning_rate=0.001, dynamic_tf=True
)

# Plot training loss
plt.plot(loss)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("plots/training_loss.png")
plt.show()

# Plot predictions
plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)

plt.close('all')
'''

# Train the model with validation
losses, val_losses = model.train_model_with_validation(
    input_tensor=X_train,
    target_tensor=Y_train,
    val_input_tensor=X_test,
    val_target_tensor=Y_test,
    n_epochs=10,
    target_len=output_window,
    batch_size=16,
    training_prediction='mixed_teacher_forcing',
    teacher_forcing_ratio=0.6,
    learning_rate=0.001,
    dynamic_tf=True
)

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("plots/training_validation_loss.png")
plt.show()

# Evaluate and plot predictions
plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)
plt.close('all')