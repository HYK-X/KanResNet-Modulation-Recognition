import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import poutyne as pt
from poutyne.framework.metrics import TopKAccuracy
from datetime import datetime
import torch_modulation_recognition as tmr
import wandb
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="训练一个调制识别模型。")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "kanresnet"], help="要训练的模型")
    parser.add_argument("--epochs", type=int, default=25, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=512, help="每个批次的样本数量（设置较低以减少CUDA内存使用）")
    parser.add_argument("--split", type=float, default=0.8, help="训练集的数据比例")
    return parser.parse_args()


def setup_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name, n_classes, dropout):
    if model_name == "resnet":
        return tmr.models.ResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    elif model_name == "kanresnet":
        return tmr.models.KanResNet(n_channels=2, n_classes=n_classes, n_res_blocks=8, n_filters=32)
    else:
        raise ValueError(f"Unknown Model: {model_name}")


def main():
    args = parse_args()
    setup_seed()
    N_CLASSES = 11
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SPLIT = args.split
    DROPOUT = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = "models"
    LOG_DIR = os.path.join("logs", args.model)


    id_str = f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb_logger = pt.WandBLogger(name=f"{id_str} run", project=f"{args.model} project")
    config_dict = {
        "Optimizer": "Adam",
        "Loss": "Cross-Entropy",
        "learning_rate": 0.01,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "split": SPLIT
    }
    wandb_logger.log_config_params(config_params=config_dict)
    print("loading model")
    net = load_model(args.model, N_CLASSES, DROPOUT)
    dataset = tmr.data.RadioML2016()


    total = len(dataset)
    lengths = [int(total * SPLIT), total - int(total * SPLIT)]
    print(f"spilit to {lengths[0]} train {lengths[1]} val")
    train_set, val_set = random_split(dataset, lengths)


    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE)


    base_save_path = Path(f"./{args.model}_model_checkpoints/{id_str}")
    base_save_path.mkdir(exist_ok=True, parents=True)

    general_callbacks = [
        pt.ProgressionCallback(),
        wandb_logger,
        pt.CSVLogger(str(base_save_path / "train_log.csv")),
    ]

    top3 = TopKAccuracy(k=3)
    top5 = TopKAccuracy(k=5)
    metrics = ["acc", top3, top5]


    experiment = pt.Experiment(
        directory=MODEL_DIR,
        network=net,
        device=DEVICE,
        optimizer="Adam",
        loss_function="cross_entropy",
        batch_metrics=metrics,
        logging=False
    )
    experiment.train(
        train_generator=train_dataloader,
        valid_generator=val_dataloader,
        epochs=EPOCHS,
        seed=42,
        callbacks=general_callbacks
    )

    torch.save(net.state_dict(), base_save_path / f"{args.model}_final_weights.pt")
    wandb.finish()


if __name__ == "__main__":
    main()

