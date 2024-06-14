import os
import gc
import itertools
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict

# 调制类型及其对应的索引
MODULATIONS = {
    "8PSK": 0,
    "BPSK": 1,
    "CPFSK": 2,
    "GFSK": 3,
    "PAM4": 4,
    "QAM16": 5,
    "QAM64": 6,
    "QPSK": 7,
    "AM-DSB": 8,
    "AM-SSB": 9,
    "WBFM": 10,
}

# 信噪比列表
SNRS = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
    -20, -18, -16, -14, -12, -10, -8, -6, -4, -2
]


class RadioML2016(torch.utils.data.Dataset):
    modulations = MODULATIONS
    snrs = SNRS

    def __init__(
            self,
            data_dir: str = "./data",
            file_name: str = "RML2016.10a_dict.pkl"
    ):
        self.file_name = file_name
        self.data_dir = data_dir
        self.n_classes = len(self.modulations)
        self.X, self.y = self.load_data()
        gc.collect()  # 垃圾回收，释放内存

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """ 从文件中加载数据 """
        print("从文件中加载数据...")
        with open(os.path.join(self.data_dir, self.file_name), "rb") as f:
            data = pickle.load(f, encoding="latin1")

        X, y = [], []
        print("处理数据集")
        for mod, snr in tqdm(list(itertools.product(self.modulations, self.snrs))):
            X.append(data[(mod, snr)])

            for i in range(data[(mod, snr)].shape[0]):
                y.append((mod, snr))

        X = np.vstack(X)  # 将X列表垂直堆叠成一个numpy数组

        return X, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 加载一个批次的输入和标签 """
        x, (mod, snr) = self.X[idx], self.y[idx]
        y = self.modulations[mod]
        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)  # 增加一个维度，使其符合模型输入
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]

    def get_signals(self, mod: List[str] = None, snr: List[int] = None) -> Dict:
        """ 返回特定调制或信噪比的信号 """

        # 如果mod或snr为None，则表示所有的调制或信噪比
        if mod is None:
            mod = list(self.modulations.keys())
        if snr is None:
            snr = self.snrs

        # 如果单个mod或snr，则转换为列表以便于迭代
        if not isinstance(mod, list):
            mod = [mod]
        if not isinstance(snr, list):
            snr = [snr]

        # 聚合信号到一个字典中
        X = {}
        for mod, snr in list(itertools.product(mod, snr)):
            X[(mod, snr)] = []
            for idx, (m, s) in enumerate(self.y):
                if m == mod and s == snr:
                    X[(mod, snr)].append(np.expand_dims(self.X[idx, ...], axis=0))

            X[(mod, snr)] = np.concatenate(X[(mod, snr)], axis=0)

        return X
