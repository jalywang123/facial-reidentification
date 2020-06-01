import sys
import torch
import argparse
import os
import torch.nn.functional as F
from torch.autograd import Variable
from siamesenetwork import ImagesDataset
from siamesenetwork import SiameseNetwork_V2 as SiameseNetwork
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--model")
res = args.parse_args()
model_name = res.model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")


class Config:
    if sys.platform == "win32":
        testing_dir = os.path.join(os.getcwd(), "images/test")  # for windows add absolute path to avoid errors
        images_dir = os.path.join(os.getcwd(), "images/dataset/compare_v2")
    else:
        testing_dir = "./images/test/"  # add linux path here before testing
        images_dir = "./compare_v2/"


print("loading image datasets... ")

test_dataset = ImagesDataset(
    rootdir=Config.testing_dir,
    transform=transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    ),
)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print("prepared test image")

compare_dataset = ImagesDataset(
    rootdir=Config.images_dir,
    transform=transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    ),
)

compare_dataloader = DataLoader(compare_dataset, batch_size=1, shuffle=False)
print("prepared compare dataset")

dataiter_test_image = iter(test_dataloader)
x0 = next(dataiter_test_image)

# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# img = Image.fromarray(np.array(x0["image"]).reshape(100, 100))
# img.show()

dataiter = iter(compare_dataloader)

net = None
loaded_model = None

print(f'Test Image: {x0["name"]}')

net = SiameseNetwork().to(device)
loaded_model = torch.load(model_name, map_location=torch.device(device))
net.load_state_dict(loaded_model["state_dict"])
if net is not None:
    print("model loaded... ")
else:
    print("model loading failed... try again")

# test_img = test_dataset[0]["image"].to(device)
dist = []
for i in range(len(compare_dataset)):
    x1 = next(dataiter)
    # _,x1,label2 = next(dataiter)
    # compare_image = compare["image"]
    # concatenated = torch.cat((x0["image"].to(device), x1["image"].to(device)), 0)

    output1, output2 = net(
        Variable(x0["image"].to(device)), Variable(x1["image"].to(device))
    )
    euclidean_distance = F.pairwise_distance(output1, output2)
    ed = euclidean_distance.cpu().detach().numpy()
    print(f"filename : {x1['name']} ed: {euclidean_distance}")
    dist.append(ed)
#print(dist, end='\n')
compare_images = list(os.listdir(Config.images_dir))
match = np.argmin(dist)
print(f'Prediction: {compare_images[match]}')
    # print(euclidean_distance)
    # if euclidean_distance <= 0.5:
    #     print(f"match found... : {x1['name']} ed: {euclidean_distance}")
    #     # torchvision.utils.save_image(concatenated, f'match.png')
    #     break
    # print(f"match not found... {x0['name']} and {x1['name']}, ed: {euclidean_distance}")
    # check distance for all outputs
    # print(f"filename : {x1['name']} ed: {euclidean_distance}")
