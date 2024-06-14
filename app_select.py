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


# Load models
models = {
    "resnet": load_model("resnet", N_CLASSES, DROPOUT, "resnet_final_weights.pt"),
    "kanresnet": load_model("kanresnet", N_CLASSES, DROPOUT, "kanresnet_final_weights.pt")
}
models["resnet"] = models["resnet"].to(DEVICE)
models["kanresnet"] = models["kanresnet"].to(DEVICE)


def visualize_signal(modulation: str, snr: int, idx: int) -> plt.Figure:
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

    return fig


def predict_signal_wrapper(model_name: str, modulation: str, snr: int, idx: int) -> str:
    signals = data.get_signals(mod=modulation, snr=snr)

    # Generate random index if idx is -1
    if idx == -1:
        idx = random.randrange(NUM_PER_CLASS)

    signal = signals[(modulation, snr)][idx]

    # Prepare signal for prediction
    signal = np.expand_dims(signal, axis=0)  # Increase dimension (1, 2, 128)
    signal = np.expand_dims(signal, axis=0)  # Further increase dimension (1, 1, 2, 128)
    signal_tensor = torch.from_numpy(signal).float().to(DEVICE)
    prediction = predict_signal(models[model_name], signal_tensor, DEVICE)
    predicted_modulation = MODULATIONS[prediction]

    return f"Predicted Modulation: {predicted_modulation}"


with gr.Blocks(title="Signal Processing Interface", theme="default") as demo:
    gr.Markdown("<h1>Signal Processing Interface</h1>")
    gr.Markdown("Select the appropriate options to visualize or predict signal properties.")
    with gr.Tab("Visualize Signal"):
        modulation_dropdown_viz = gr.Dropdown(choices=MODULATIONS, label="Modulation")
        snr_dropdown_viz = gr.Dropdown(choices=SNRS, label="Signal-to-Noise Ratio")
        idx_slider_viz = gr.Slider(minimum=-1, maximum=NUM_PER_CLASS - 1, label="Index (set to -1 for random)")
        visualize_button = gr.Button("Visualize Signal")
        visualize_plot = gr.Plot()

        visualize_button.click(fn=visualize_signal,
                               inputs=[modulation_dropdown_viz, snr_dropdown_viz, idx_slider_viz],
                               outputs=visualize_plot)

    with gr.Tab("Predict Signal"):
        model_dropdown = gr.Dropdown(choices=["resnet", "kanresnet"], label="Model")
        modulation_dropdown_pred = gr.Dropdown(choices=MODULATIONS, label="Modulation")
        snr_dropdown_pred = gr.Dropdown(choices=SNRS, label="Signal-to-Noise Ratio")
        idx_slider_pred = gr.Slider(minimum=-1, maximum=NUM_PER_CLASS - 1, label="Index (set to -1 for random)")
        predict_button = gr.Button("Predict Signal")
        prediction_output = gr.Textbox()

        predict_button.click(fn=predict_signal_wrapper,
                             inputs=[model_dropdown, modulation_dropdown_pred, snr_dropdown_pred, idx_slider_pred],
                             outputs=prediction_output)

if __name__ == "__main__":
    demo.launch()
