from architecture.transformer import Transformer, PositionalEncoding, InputFeedForward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset_for_transformer import TrainDataset, ValidDataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import matplotlib.pyplot as plt
import numpy as np

model = Transformer(d_model=1024, num_heads=8, num_layers=3, d_ff=4096, dropout=0.1)
pe = PositionalEncoding(d_model=1024, max_seq_length=40)
input_embedding = InputFeedForward()
output_embedding = InputFeedForward()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# load dataset
data_root = '../dataset'
patch_size = 32
stride = 16
batch_size=20
print("\nloading dataset ...")
train_data = TrainDataset(data_root=data_root, crop_size=patch_size, bgr2rgb=True, arg=True, stride=stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
end_epoch = 10
per_epoch_iteration = len(train_data)
total_iteration = per_epoch_iteration*end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

init_lr=4e-4
optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)


iteration=0
outf = "exp/mst_experiment/"
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

record_mrae_loss = 1000

while iteration<total_iteration:
    model.train()
    losses = AverageMeter()
    for i, (images) in enumerate(train_loader):                        
        images = pe(images)   # add positional encoding

        rgb_image = images[:, :3, :]
        rgb_image = input_embedding(rgb_image)

        hsi_input = images[:, 3:6, :]
        hsi_input = output_embedding(hsi_input)

        labels = images[:, 6, :]
        lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        output = model(rgb_image, hsi_input)
        loss = criterion_rmse(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.data)
        iteration = iteration+1
        if iteration % 20 == 0:
            print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f' % (iteration, total_iteration, lr, losses.avg))
