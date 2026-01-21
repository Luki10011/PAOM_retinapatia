import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torchmetrics
from torch.utils.tensorboard import SummaryWriter


# --------------------------
# Blok rezydualny
# --------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class DeepCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # -------- Block 1 --------
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 224 → 112

            # -------- Block 2 --------
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 112 → 56

            # -------- Block 3 --------
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 56 → 28

            # -------- Block 4 --------
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 28 → 14
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x

class RGBResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.path_R = self._make_path()
        self.path_G = self._make_path()
        self.path_B = self._make_path()

        # Fully connected po GAP
        self.fc = nn.Sequential(
            nn.Linear(64*3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Dropout(0.5)
        )

    def _make_path(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=2)
        )

    def forward(self, x):
        R = x[:,0:1,:,:]
        G = x[:,1:2,:,:]
        B = x[:,2:3,:,:]

        out_R = self.path_R(R)
        out_G = self.path_G(G)
        out_B = self.path_B(B)

        out_R = F.adaptive_avg_pool2d(out_R, (1,1)).view(x.size(0), -1)
        out_G = F.adaptive_avg_pool2d(out_G, (1,1)).view(x.size(0), -1)
        out_B = F.adaptive_avg_pool2d(out_B, (1,1)).view(x.size(0), -1)

        out = torch.cat([out_R, out_G, out_B], dim=1)

        # Fully connected
        out = self.fc(out)
        return out

class DRLightning(LightningModule):
    def __init__(self, in_channels=3, num_classes=5, learning_rate=1e-3, img_size=224):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.img_size = img_size

        self.save_hyperparameters()

        self.model = RGBResNet(num_classes=num_classes)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.train_acc(logits.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(logits.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.test_acc(logits.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate) # type: ignore
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        return {
                'optimizer': opt,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'
                }
    

def visualize_model():
    model = DeepCNN(num_classes=5)
    model.eval()
    print(model)
    # Dummy input: (batch, channels, H, W)
    dummy_input = torch.randn(1, 3, 224, 224)

    writer = SummaryWriter(log_dir="runs/RGBResNet")

    # Najważniejsza linia
    writer.add_graph(model, dummy_input)

    writer.close()
    print("Graf zapisany. Uruchom TensorBoard.")

if __name__ == "__main__":
    visualize_model()
