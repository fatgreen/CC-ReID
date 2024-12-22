# import os
# import time
# import datetime
# import logging
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import distributed as dist
# import torchvision
# from torchvision import datasets, models, transforms
# from configs.default_img import get_img_config
# from models.img_resnet import ResNet50
# from models.resnet101 import ResNet101
# from models.resnet152 import ResNet152
# from PIL import Image
# import matplotlib.pyplot as plt


# def parse_option():
#     parser = argparse.ArgumentParser(
#         description='Train clothes-changing re-id model with clothes-based adversarial loss')
#     parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
#     # Datasets
#     parser.add_argument('--root', type=str, help="your root path to data directory")
#     # Miscs
#     parser.add_argument('--img_folder', type=str, help='path to the folder containing images')
#     parser.add_argument('--weights', type=str, help='path to the weights')
#     parser.add_argument('--gpu', type=str, default='0', help='gpu id')

#     args, unparsed = parser.parse_known_args()
#     config = get_img_config(args)
#     return config, args


# @torch.no_grad()
# def extract_img_feature(model, img):
#     flip_img = torch.flip(img, [3])
#     img, flip_img = img.cuda(), flip_img.cuda()
#     _, batch_features = model(img)
#     _, batch_features_flip = model(flip_img)
#     batch_features += batch_features_flip
#     batch_features = F.normalize(batch_features, p=2, dim=1)
#     features = batch_features.cpu()

#     return features


# def generate_heatmap(feature, image_size):
#     # Normalize feature to [0, 1]
#     feature_min, feature_max = feature.min(), feature.max()
#     normalized_feature = (feature - feature_min) / (feature_max - feature_min)

#     # Reshape feature to match smaller spatial dimensions if necessary
#     feature_size = int(np.sqrt(feature.size(0)))  # Assume feature is square
#     reshaped_feature = normalized_feature.numpy().reshape(feature_size, feature_size)

#     # Resize heatmap to match original image dimensions
#     heatmap = np.array(Image.fromarray((reshaped_feature * 255).astype(np.uint8)).resize(image_size, Image.BILINEAR))

#     return heatmap


# def overlay_heatmaps(images, features):
#     overlays = []
#     for image, feature in zip(images, features):
#         heatmap = generate_heatmap(feature, image.size)
#         overlays.append((image, heatmap))
#     return overlays


# config, args = parse_option()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# dict = torch.load(args.weights)
# model = ResNet152(config)
# model.load_state_dict(dict['model_state_dict'])
# model = model.cuda()
# model.eval()

# # Data transforms
# data_transforms = transforms.Compose([
#         transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # Load and preprocess images from folder
# image_folder = args.img_folder
# image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
# images = [Image.open(path) for path in image_paths]
# image_tensors = [data_transforms(image).unsqueeze(0) for image in images]
# input_batches = torch.cat(image_tensors, dim=0)  # Combine all images into a single batch

# # Extract features
# features = extract_img_feature(model, input_batches)

# # Generate overlays
# overlays = overlay_heatmaps(images, features)

# # Visualize results
# plt.figure(figsize=(12 * len(images), 6))
# for i, (image, heatmap) in enumerate(overlays):
#     # Display original image
#     plt.subplot(2, len(images), i + 1)
#     plt.imshow(image)
#     plt.axis("off")

#     # Display heatmap
#     plt.subplot(2, len(images), len(images) + i + 1)
#     plt.imshow(heatmap, cmap='jet')
#     plt.colorbar()
#     plt.axis("off")

# plt.show()


import os
import time
import datetime
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
import torchvision
from torchvision import datasets, models, transforms
from configs.default_img import get_img_config
from models.img_resnet import ResNet50
from PIL import Image
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    # Miscs
    parser.add_argument('--img_path', type=str, help='path to the image')
    parser.add_argument('--weights', type=str, help='path to the weights')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args)
    return config, args

@torch.no_grad()
def extract_img_feature(model, img):
    flip_img = torch.flip(img, [3])
    img, flip_img = img.cuda(), flip_img.cuda()
    _, batch_features = model(img)
    _, batch_features_flip = model(flip_img)
    batch_features += batch_features_flip
    batch_features = F.normalize(batch_features, p=2, dim=1)
    features = batch_features.cpu()

    return features

def generate_heatmap(feature, image_size):
    # Normalize feature to [0, 1]
    feature_min, feature_max = feature.min(), feature.max()
    normalized_feature = (feature - feature_min) / (feature_max - feature_min)

    # Reshape feature to match smaller spatial dimensions if necessary
    feature_size = int(np.sqrt(feature.size(0)))  # Assume feature is square
    reshaped_feature = normalized_feature.numpy().reshape(feature_size, feature_size)

    # Resize heatmap to match original image dimensions
    heatmap = np.array(Image.fromarray((reshaped_feature * 255).astype(np.uint8)).resize(image_size, Image.BILINEAR))

    return heatmap

config, args = parse_option()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dict = torch.load(args.weights)
model = ResNet50(config)
model.load_state_dict(dict['model_state_dict'], strict=False)  # Allow mismatched keys
model = model.cuda()
model.eval()

# Data transforms
data_transforms = transforms.Compose([
        transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open(args.img_path)
image_tensor = data_transforms(image)
input_batch = image_tensor.unsqueeze(0)  # Add a batch dimension

# Extract features
feature = extract_img_feature(model, input_batch)

# Generate heatmap
image_size = image.size
heatmap = generate_heatmap(feature[0], image_size)

# Overlay heatmap on original image
heatmap_color = plt.cm.jet(heatmap / 255.0)[:, :, :3]  # Get RGB from colormap
original_image = np.array(image) / 255.0  # Normalize original image to [0, 1]

# Combine heatmap and original image
overlay_image = (0.5 * original_image + 0.5 * heatmap_color)
overlay_image = np.clip(overlay_image, 0, 1)  # Ensure values are within [0, 1]

# Visualize original image, heatmap, and overlay
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Heatmap
plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='jet')
plt.title("Heatmap")
plt.colorbar()
plt.axis("off")

# Overlay image
plt.subplot(1, 3, 3)
plt.imshow(overlay_image)
plt.title("Overlay")
plt.axis("off")

plt.show()
