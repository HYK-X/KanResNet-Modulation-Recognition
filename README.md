# KanResNet-Modulation-Recognition
This project compares the performance of KanResNet and ResNet in recognizing modulation schemes using the RadioML2016.10a dataset. It also provides an interactive interface for dataset exploration and signal prediction.
## Install Requirements

```bash
pip install -r requirements.txt
pip install poutyne
pip install gradio
```

## Train

```bash
# Train ResNet
python train.py --model resnet --epochs 25 --batch_size 512 --split 0.8

# Train KanResNet
python train.py --model kanresnet --epochs 25 --batch_size 512 --split 0.8

```

### Results

After training Wandb logs will be located in the logs directory .

## Evaluate
```bash
python evaluate.py --models resnet kanresnet --weights resnet_final_weights.pt kanresnet_final_weights.pt
```

This project is based on the [project]( https://github.com/isaaccorley/pytorch-modulation-recognition)