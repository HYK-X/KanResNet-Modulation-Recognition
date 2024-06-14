import torch
import torch.nn as nn
from torch_modulation_recognition.fast_kan_conv import FastKANConv1DLayer

class ResidualBlock(nn.Module):
    """Base Residual Block
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation
    ):

        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            activation(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x) + x
        return x


class KanResidualBlock(nn.Module):
    """Base Residual Block
    """

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            activation
    ):
        super(KanResidualBlock, self).__init__()

        self.model = nn.Sequential(
            FastKANConv1DLayer(input_dim=channels, output_dim=channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2),
            activation(),
            FastKANConv1DLayer(input_dim=channels, output_dim=channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x) + x
        return x


class KanResNet(torch.nn.Module):
    """Modulation Recognition ResNet (Mine)"""

    def __init__(
            self,
            n_channels: int = 2,
            n_classes: int = 10,
            n_res_blocks: int = 16,
            n_filters: int = 32,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(KanResNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_res_blocks = n_res_blocks
        self.device = device
        self.loss = nn.CrossEntropyLoss()

        self.head = nn.Sequential(
            FastKANConv1DLayer(input_dim=self.n_channels, output_dim=self.n_filters, kernel_size=3, stride=1,
                               padding=4),
            nn.ReLU()
        )

        # Residual Blocks
        self.res_blocks = [
            KanResidualBlock(channels=self.n_filters, kernel_size=3, activation=nn.ReLU) \
            for _ in range(self.n_res_blocks)
        ]
        self.res_blocks.append(
            FastKANConv1DLayer(input_dim=self.n_filters, output_dim=self.n_filters, kernel_size=3, stride=1, padding=1))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        # Output layer
        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_filters, out_features=self.n_filters, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.n_filters, out_features=n_classes, bias=True),
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = self.head(x)
        shortcut = x
        x = self.res_blocks(x) + shortcut

        # Global average pooling
        x = torch.mean(x, dim=-1)

        # Classification
        x = self.tail(x)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = x.to(self.device)
        y_pred = self.forward(x)
        y_pred = y_pred.to("cpu")
        y_pred = torch.softmax(y_pred, dim=-1)
        values, indices = torch.max(y_pred, dim=-1)
        indices = indices.numpy()
        return indices


class ResNet(torch.nn.Module):
    """Modulation Recognition ResNet (Mine)"""

    def __init__(
            self,
            n_channels: int = 2,
            n_classes: int = 11,
            n_res_blocks: int = 8,
            n_filters: int = 32,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(ResNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_res_blocks = n_res_blocks
        self.device = device
        self.loss = nn.CrossEntropyLoss()

        self.head = nn.Sequential(
            nn.Conv1d(in_channels=self.n_channels, out_channels=self.n_filters, kernel_size=3, stride=1, padding=4),
            nn.ReLU()
        )

        # Residual Blocks
        self.res_blocks = [
            ResidualBlock(channels=self.n_filters, kernel_size=3, activation=nn.ReLU) \
            for _ in range(self.n_res_blocks)
        ]
        self.res_blocks.append(
            nn.Conv1d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=3, stride=1, padding=1))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        # Output layer
        self.tail = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_filters, out_features=self.n_filters, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.n_filters, out_features=n_classes, bias=True),
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = self.head(x)
        shortcut = x
        x = self.res_blocks(x) + shortcut
        # Global average pooling
        x = torch.mean(x, dim=-1)
        # Classification
        x = self.tail(x)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = x.to(self.device)
        y_pred = self.forward(x)
        y_pred = y_pred.to("cpu")
        y_pred = torch.softmax(y_pred, dim=-1)
        values, indices = torch.max(y_pred, dim=-1)
        indices = indices.numpy()
        return indices
