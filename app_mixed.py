import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch_modulation_recognition as tmr
from typing import Tuple

NUM_PER_CLASS = 1000
N_CLASSES = 11
DROPOUT = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
data = tmr.data.RadioML2016()
MODULATIONS = list(tmr.data.MODULATIONS.keys())
SNRS = tmr.data.SNRS

# Custom Dataset class
class SignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray = None):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.signals[idx]
        y = -1 if self.labels is None else self.labels[idx]
        x = torch.from_numpy(x).float()
        return x, y

# Model loading function
def load_model(model_name: str, n_classes: int, dropout: float, weights_path: str):
    if model_name == "resnet":
        model = tmr.models.ResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    elif model_name == "kanresnet":
        model = tmr.models.KanResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model

# Prediction function
def predict_signal(model, signal, device):
    model.eval()
    with torch.no_grad():
        signal = signal.to(device)
        output = model(signal)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Load model
model = load_model("resnet", N_CLASSES, DROPOUT, "resnet_final_weights.pt")
model = model.to(DEVICE)

def visualize_and_predict(modulation: str, snr: int, idx: int) -> Tuple[plt.Figure, str]:
    signals = data.get_signals(mod=modulation, snr=snr)

    # Generate random index if idx is -1
    if idx == -1:
        idx = random.randrange(NUM_PER_CLASS)

    signal = signals[(modulation, snr)][idx]

    # Plot signal
    fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
    plt.xlabel("time (s)")
    fig.suptitle(
        "Index: {}, Modulation: {}, SNR: {} (In-Phase (I) and Quadrature (Q) Channels)".format(idx, modulation, snr))
    ax[0].plot(signal[0, :])
    ax[1].plot(signal[1, :])

    # Prepare signal for prediction
    signal = np.expand_dims(signal, axis=0)  # Increase dimension (1, 2, 128)
    signal = np.expand_dims(signal, axis=0)  # Further increase dimension (1, 1, 2, 128)
    signal_tensor = torch.from_numpy(signal).float().to(DEVICE)
    prediction = predict_signal(model, signal_tensor, DEVICE)
    predicted_modulation = MODULATIONS[prediction]

    return fig, f"Predicted Modulation: {predicted_modulation}"

# Create Gradio interface
modulation_dropdown = gr.Dropdown(choices=MODULATIONS, label="Modulation")
snr_dropdown = gr.Dropdown(choices=SNRS, label="Signal-to-Noise Ratio")
idx_slider = gr.Slider(minimum=-1, maximum=NUM_PER_CLASS - 1, label="Index (set to -1 for random)")

combined_iface = gr.Interface(fn=visualize_and_predict,
                              inputs=[modulation_dropdown, snr_dropdown, idx_slider],
                              outputs=["plot", "text"],
                              title="Signal Visualization and Prediction",
                              description="Select modulation, SNR, and index to visualize the signal and predict its modulation.")

if __name__ == "__main__":
    combined_iface.launch()
