import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

device = torch.device('mps')

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((256, 256))
        mask = mask.resize((256, 256))

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


transform = A.Compose(
        [A.Resize(height=160, width=160),
         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ToTensorV2()])

dataset = LungDataset(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        transform=transform
        )

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
        )

model = model.to(device)

criterion = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    loss_counter = 0
    for i, (images, masks) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        loss_counter += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/loss_counter}")

torch.save(model.state_dict(), 'model.pth')
