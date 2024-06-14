import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch_modulation_recognition as tmr
from pathlib import Path
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
# 模型加载函数
def load_model(model_name, n_classes, dropout, weights_path):
    if model_name == "resnet":
        model = tmr.models.ResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    elif model_name == "kanresnet":
        model = tmr.models.KanResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    else:
        raise ValueError(f"未知的模型: {model_name}")

    model.load_state_dict(torch.load(weights_path))
    return model


# 自定义数据集类
class SignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.signals[idx]
        y = self.labels[idx]
        x = torch.from_numpy(x).float().unsqueeze(0)  # 增加一个维度，使其符合模型输入
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# 评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


# 根据SNR划分数据集并评估
def evaluate_by_snr(model, dataset, device):
    results = {}
    snrs = dataset.snrs

    for snr in snrs:
        signals_dict = dataset.get_signals(snr=[snr])
        signals, labels = [], []
        for (mod, _), mod_signals in signals_dict.items():
            signals.append(mod_signals)
            labels.append(np.full(mod_signals.shape[0], dataset.modulations[mod]))

        signals = np.vstack(signals)
        labels = np.concatenate(labels)

        snr_dataset = SignalDataset(signals, labels)
        snr_dataloader = DataLoader(snr_dataset, batch_size=512)
        accuracy = evaluate_model(model, snr_dataloader, device)
        results[snr] = accuracy
        print(f"SNR: {snr}, Accuracy: {accuracy:.4f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="评估调制识别模型。")
    parser.add_argument("--models", type=str, nargs='+', required=True, choices=["resnet", "kanresnet"], help="要评估的模型")
    parser.add_argument("--weights", type=str, nargs='+', required=True, help="模型权重的路径")
    parser.add_argument("--batch_size", type=int, default=512, help="每个批次的样本数量")
    return parser.parse_args()


def main():
    args = parse_args()

    # 参数设置
    N_CLASSES = 11
    DROPOUT = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    dataset = tmr.data.RadioML2016()

    # 评估每个模型
    results = {}

    for model_name, weights_path in zip(args.models, args.weights):

        print(f"加载模型: {model_name}")
        model = load_model(model_name, N_CLASSES, DROPOUT, weights_path)
        model = model.to(DEVICE)

        print(f"评估模型: {model_name}")
        model_results = evaluate_by_snr(model, dataset, DEVICE)
        results[model_name] = model_results
    model_name_mapping = {
        "resnet": "ResNet",
        "kanresnet": "KanResNet"
    }
    # 打印结果并绘制图形
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        # 使用映射字典获取正确的模型名称
        plotted_model_name = model_name_mapping.get(model_name, model_name)
        snrs = sorted(model_results.keys())
        accuracies = [model_results[snr] for snr in snrs]
        plt.plot(snrs, accuracies, marker='o', linestyle='-', label=plotted_model_name)

    plt.xlabel('SNR')
    plt.xticks(range(min(snrs), max(snrs) + 1, 2))
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('model_comparison_accuracy_vs_snr.png')  # 保存图表
    plt.show()  # 显示图表


    root = ET.Element("results")

    for model_name, model_results in results.items():
        model_elem = ET.SubElement(root, "model")
        model_elem.set("name", model_name)

        for snr, accuracy in model_results.items():
            result_elem = ET.SubElement(model_elem, "result")
            result_elem.set("snr", str(snr))
            result_elem.text = f"{accuracy:.4f}"

    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    xml_str_pretty = minidom.parseString(xml_str).toprettyxml(indent="   ")
    with open("results.xml", "w", encoding="utf-8") as f:
        f.write(xml_str_pretty)


if __name__ == "__main__":
    main()
