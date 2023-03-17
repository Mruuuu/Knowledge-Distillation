'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-10 16:00:44
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-03-17 23:53:52
FilePath: /mru/Knowledge-Distillation/predict.py
Description: 

'''
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torchvision.models as models
from torchsummary import summary
import pandas as pd

# from model import resnet14


def test_resnet18_on_fashion_mnist(model_path):
    
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # transform
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3),  # gray to 3 channel
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # load Fashion MNIST
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)

    # load the pruned model
    net = torch.load(model_path)
    net.eval()

    # print summary
    summary(net.cuda(), (3, 28, 28))
    params = sum(p.numel() for p in net.parameters())
    print(f"Number of remaining parameters: {params}")

    # test 
    net = net.to(device)
    correct = 0
    total = 0
    pred_arr = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_arr.append(predicted.item())

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")

    pred_data = {"pred":pred_arr}
    df_pred = pd.DataFrame(pred_data)
    df_pred.to_csv('pred.csv', index_label='id')

    return accuracy

def main():

    model_path = sys.argv[1]
    test_resnet18_on_fashion_mnist(model_path)
    

if __name__ == "__main__":
    main()        
