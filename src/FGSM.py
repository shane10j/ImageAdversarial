import torch
import torch.nn as nn
from torchvision import transforms, models, datasets

from keras.datasets import mnist
import timm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resnet_34(device):
    model = models.resnet18(pretrained=True)
    model.to(device)
    model.eval()
    return model

def load_vit_base_patch16_224(device):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.to(device)
    return model


def get_dataloaders(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, epochs=5, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_correct = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            num_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = num_correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    model.eval()
    return model


def fgsm_attack(image, epsilon, data_gradient):
    sign_gradient = data_gradient.sign()
    perturbed = image + epsilon*sign_gradient
    perturbed = torch.clamp(perturbed, 0, 1) #Guarantees all values are betewen 0 and 1, since image is normalized
    return perturbed

def execute_fgsm(model, device, test_loader, epsilon, num_images=5):
    #Load Data
    image_count = 0
    incorrectly_classified = 0
    images_tried = 0
    initial_images = []
    perturbed_images = []
    corresponding_labels = []
    for images, labels in test_loader:
        images_tried += 1
        images, labels = images.to(device), labels.to(device)
        _, init_pred = model(images).max(1)
        if init_pred == labels:
            print("Image initially classified correctly as: ", init_pred.item())
            image_count += 1
            initial_images.append(images)
            images.requires_grad = True
            criterion = nn.CrossEntropyLoss()
            outputs = model(images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_gradient = images.grad.data
            perturbed = fgsm_attack(images, epsilon, data_gradient)
            perturbed_images.append(perturbed)
            adv_outputs = model(perturbed)
            _, adv_pred = adv_outputs.max(1)
            print("Adversarial image classified as: ", adv_pred.item())
            if adv_pred != init_pred:
                incorrectly_classified += 1
            corresponding_labels.append((labels, adv_pred))
        if image_count == num_images:
            break
    print("Number of images tried: ", images_tried)
    try:
        print("Percentage of perturbed images incorrectly classified: ", (incorrectly_classified/image_count)*100)
    except ZeroDivisionError:
        print("Model did not correctly classify any images")
    return initial_images, perturbed_images, corresponding_labels

def visualize(orig, perturbed):
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    #orig_disp = inv_norm(orig.squeeze()).permute(1,2,0).cpu().detach().numpy()
    #pert_disp = inv_norm(perturbed.squeeze()).permute(1,2,0).cpu().detach().numpy()
    orig_disp = orig.squeeze().permute(1,2,0).cpu().detach().numpy()
    pert_disp = perturbed.squeeze().permute(1,2,0).cpu().detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.imshow(orig_disp)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(pert_disp)
    ax2.set_title('Perturbed')
    ax2.axis('off')
    plt.show()
