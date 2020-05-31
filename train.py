import sys
import time
import torch
import PIL.ImageOps
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torchvision.utils
from helpers import show_plot
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.datasets.fakedata
import torchvision.transforms as transforms
from loss import ContrastiveLoss
from siamesenetwork import SiameseNetwork
from siamesenetwork import SiameseNetworkDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")


class Config:
    training_dir = "./images/train_q/"
    train_batch_size = 32
    train_number_epochs = 50_000


folder_dataset = dset.ImageFolder(root=Config.training_dir)

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    should_invert=False,
)

train_dataloader = DataLoader(
    siamese_dataset,
    shuffle=True,
    # num_workers=8,
    batch_size=Config.train_batch_size,
)

net = SiameseNetwork().to(device)
criterion = ContrastiveLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0003)

counter = []
loss_history = []
iteration_number = 0

try:
    import slack_callback
except Exception as e:
    pass

for epoch in range(1, Config.train_number_epochs + 1):
    start = time.time()
    for i, data in enumerate(
        train_dataloader, 0
    ):
        img0, img1, label = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("epoch: {} current loss: {} time taken: {}s".format(epoch, loss_contrastive.item(), str(time.time() - start)[:5]))
            message = "epoch: {} current loss: {} time taken: {}s".format(
                                        epoch, loss_contrastive.item(), str(time.time() - start)[:5]
                                                        )
            try:
               slack_callback.write(message)
            except Exception as e:
               pass

            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    if epoch % 50 == 0 and epoch > 0:
        state_dict = {
            "epoch": epoch + 1,
            "state_dict": net.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "counter": counter,
        }
        torch.save(state_dict, f"models/model_state_dict_{epoch}.pt")
        print()
        print(f"model checkpoint saved to models/model_state_dict_{epoch}")

show_plot(counter, loss_history, save=True)
