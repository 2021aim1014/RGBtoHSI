import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import datetime
from hsi_dataset import TrainDataset, ValidDataset
from architecture import model_generator

# Configuration
CONFIG = {
    "method": "mirnet",
    "pretrained_model_path": None,
    "batch_size": 2,
    "epochs": 100,
    "learning_rate": 4e-4,
    "output_dir": "./exp/mst_plus_plus/",
    "data_root": "dataset/pear dataset/train_dataset/",
    "val_data_root": "dataset/pear dataset/val_dataset/",
    "patch_size": 128,
    "stride": 128,
    "gpu_id": "0",
    "per_epoch_iteration": 1,
}
CONFIG["total_iteration"] = CONFIG["per_epoch_iteration"] * CONFIG["epochs"]

# Setup CUDA
torch.cuda.set_device(int(CONFIG["gpu_id"])) if torch.cuda.is_available() else None

# Load dataset
print("\nLoading dataset...")
train_data = TrainDataset(
    data_root=CONFIG["data_root"], crop_size=CONFIG["patch_size"], bgr2rgb=True, arg=True, stride=CONFIG["stride"]
)
val_data = ValidDataset(data_root=CONFIG["val_data_root"], bgr2rgb=True)
print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

# Model setup
model = model_generator(CONFIG["method"], CONFIG["pretrained_model_path"])
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
if torch.cuda.is_available():
    model = model.cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Optimizer & Scheduler
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG["total_iteration"], eta_min=1e-6)

# Resume from checkpoint
if CONFIG["pretrained_model_path"] and os.path.isfile(CONFIG["pretrained_model_path"]):
    print(f"Loading checkpoint: {CONFIG['pretrained_model_path']}")
    checkpoint = torch.load(CONFIG["pretrained_model_path"])
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Validation function
def validate():
    model.eval()
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            # images, labels = map(lambda x: x.cuda(), [images, labels])
            output = model(images)
            loss = torch.nn.functional.mse_loss(output, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.9f}")
    return avg_loss

# Training function
def train():
    cudnn.benchmark = True
    iteration = 0
    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    while iteration < CONFIG["total_iteration"]:
        model.train()
        for images, labels in train_loader:
            # images, labels = map(lambda x: Variable(x.cuda()), [images, labels])
            optimizer.zero_grad()
            output = model(images)
            loss = torch.nn.functional.mse_loss(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1
            
            # if iteration % 20 == 0:
            print(f"[Iter:{iteration}/{CONFIG['total_iteration']}], lr={optimizer.param_groups[0]['lr']:.9f}, train_loss={loss.item():.9f}")
            
            if iteration % 10 == 0:
                validate()

if __name__ == "__main__":
    train()
