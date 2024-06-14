import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import torch_modulation_recognition as tmr
import random


NUM_PER_CLASS = 1000
N_CLASSES = 11
DROPOUT = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


data = tmr.data.RadioML2016()
MODULATIONS = list(tmr.data.MODULATIONS.keys())
SNRS = tmr.data.SNRS


def load_model(model_name: str, n_classes: int, dropout: float, weights_path: str):
    if model_name == "resnet":
        model = tmr.models.ResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    elif model_name == "kanresnet":
        model = tmr.models.KanResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model


class SignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.signals[idx]
        y = self.labels[idx]
        x = torch.from_numpy(x).float().unsqueeze(0)  # Increase dimension for model input
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def predict_signal(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for signal, _ in dataloader:
            print(signal.shape)
            signal = signal.to(device)
            output = model(signal)
            print(f"Output shape: {output.shape}")
            _, predicted = torch.max(output.data, 1)
            return predicted.item()

# Load model
model = load_model("resnet", N_CLASSES, DROPOUT, "resnet_final_weights.pt")
model = model.to(DEVICE)

def predict_signal_wrapper(modulation: str, snr: int, idx: int) -> str:
    signals_dict = data.get_signals(mod=[modulation], snr=[snr])
    signals = signals_dict[(modulation, snr)]
    labels = np.zeros(len(signals))

    if idx == -1:
        idx = random.randrange(NUM_PER_CLASS)

    selected_signal = signals[idx].reshape(1, 2, 128)
    signal_dataset = SignalDataset(selected_signal, labels[:1])
    signal_dataloader = DataLoader(signal_dataset, batch_size=1, shuffle=False)


    print(f"Signal shape: {selected_signal.shape}")
    prediction = predict_signal(model, signal_dataloader, DEVICE)
    predicted_modulation = MODULATIONS[prediction]

    return f"Predicted Modulation: {predicted_modulation}"

# Test function
def test_predict_signal_wrapper():
    modulation = "QPSK"
    snr = 10
    idx = 0

    result = predict_signal_wrapper(modulation, snr, idx)
    print(result)

if __name__ == "__main__":
    test_predict_signal_wrapper()
