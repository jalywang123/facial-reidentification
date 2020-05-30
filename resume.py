import sys
import time
import torch
import argparse
import PIL.ImageOps
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torchvision.utils
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.datasets.fakedata
import torchvision.transforms as transforms
from loss import ContrastiveLoss
from torch.utils.data import DataLoader
from siamesenetwork import SiameseNetwork
from siamesenetwork import SiameseNetworkDataset


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig("loss_figure.png")
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")

args = argparse.ArgumentParser()
args.add_argument("--model")
res = args.parse_args()
model_name = res.model

net = None
loaded_model = None

net = SiameseNetwork().to(device)
loaded_model = torch.load(model_name, map_location=torch.device("cuda"))
net.load_state_dict(loaded_model["state_dict"])

criterion = ContrastiveLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

optimizer.load_state_dict(loaded_model["optim_dict"])
epochs = loaded_model["epoch"]
loss_history = loaded_model["loss_history"]
counter = loaded_model["counter"]

if net is not None:
    print("model loaded... ")
else:
    print("model loading failed... try again")

print(f"resuming training from: {model_name}")


class Config:
    training_dir = "./images/train_q/"
    train_batch_size = 64
    train_number_epochs = 5000


folder_dataset = dset.ImageFolder(root=Config.training_dir)

siamese_dataset = SiameseNetworkDataset(
    imageFolderDataset=folder_dataset,
    transform=transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
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

iteration_number = counter[-1]

for epoch in tqdm(
    range(epochs, Config.train_number_epochs + epochs),
    ascii=True if sys.platform == "win32" else False,
):
    for i, data in enumerate(train_dataloader, 0):
        start = time.time()
        img0, img1, label = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print(
                "epoch: {} current loss: {} time taken: {}s\n".format(
                    epoch + 1, loss_contrastive.item(), str(time.time() - start)[:5]
                )
            )
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    if epoch % 500 == 0 and epoch > 0:
        state_dict = {
            "epoch": epoch + 1,
            "state_dict": net.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "counter": counter,
        }
        torch.save(state_dict, f"model_state_dict_{epoch}.pt")
        print()
        print(f"model checkpoint saved to models/model_state_dict_{epoch    }")

show_plot(counter, loss_history)
