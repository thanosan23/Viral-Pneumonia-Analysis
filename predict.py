import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

device = torch.device('mps')

test_image_num = 1

model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=3,                  
    classes=1
)

model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()

image = Image.open(f'dataset/images/Viral Pneumonia-{test_image_num}.png').convert("RGB")
image = image.resize((256, 256))
image = np.array(image)

transform = A.Compose(
    [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
     ToTensorV2()]
)

image = transform(image=image)['image']
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    output = torch.sigmoid(output)
    output = (output > 0.5).float() 

mask = Image.open(f'dataset/masks/Viral Pneumonia-{test_image_num}.png').convert("RGB")
mask = mask.resize((256, 256))
mask = np.array(mask)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
plt.imshow(mask, cmap='jet', alpha=0.5)
plt.title('Original Image with Actual Mask')

plt.subplot(1, 2, 2)
plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
plt.imshow(output.cpu().squeeze(), cmap='hot', alpha=0.5)
plt.title('Original Image with Predicted Mask')

plt.show()