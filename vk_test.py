import kagglehub
import os
import random
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import gc

def create_image_annotation_dataset(root_dir):
    """
    Creates a dataset mapping image paths to their corresponding annotation paths.
    Args:
        root_dir (str): Root directory containing images and annotation files.
    Returns:
        pd.DataFrame: DataFrame with ImagePath and AnnotationPath columns.
    """
    dataset = []
    for parent_dir, _, files in os.walk(root_dir):
        images = {os.path.splitext(file)[0]: file for file in files if file.endswith('.jpg')}
        annotations = {os.path.splitext(file)[0]: file for file in files if file.endswith('.xml')}
        for base_name, image_file in images.items():
            annotation_file = annotations.get(base_name)
            dataset.append({
                "ImagePath": os.path.join(parent_dir, image_file),
                "AnnotationPath": os.path.join(parent_dir, annotation_file) if annotation_file else None
            })
    return pd.DataFrame(dataset)

path = kagglehub.dataset_download("lyly99/logodet3k")
root_directory = "/LogoDet-3K"
image_annotation_df = create_image_annotation_dataset(path + root_directory)

class LogoDatasetFromDF(Dataset):
    """
    Custom PyTorch Dataset to load images and their corresponding annotations.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def parse_annotation(self, ann_file):
        tree = ET.parse(ann_file)
        root = tree.getroot()
        objects = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
        return objects

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["ImagePath"]
        ann_path = self.df.iloc[idx]["AnnotationPath"]
        image = Image.open(img_path).convert("RGB")
        annotations = self.parse_annotation(ann_path)
        if self.transform:
            image = self.transform(image)
        return {"image": image, "annotations": annotations, "image_path": img_path}

def custom_collate_fn(batch):
    """
    Custom collate function to stack images and preserve annotations in a DataLoader.
    Args:
        batch (list): List of samples from the Dataset.
    Returns:
        dict: Dictionary containing stacked images, annotations, and image paths.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    annotations = [item["annotations"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    return {"image": images, "annotations": annotations, "image_paths": image_paths}

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = LogoDatasetFromDF(image_annotation_df, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, collate_fn=custom_collate_fn)

def parse_annotation_labels(annotation_path):
    """
    Extracts object labels from an XML annotation file.
    Args:
        annotation_path (str): Path to the XML file.
    Returns:
        list: List of labels found in the annotation.
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    labels = [obj.find("name").text for obj in root.findall("object")]
    return labels

all_labels = []
for ann_path in image_annotation_df["AnnotationPath"]:
    labels = parse_annotation_labels(ann_path)
    all_labels.extend(labels)

label_distribution = Counter(all_labels)
label_distribution_df = pd.DataFrame.from_dict(label_distribution, orient='index', columns=['Count']).reset_index()
label_distribution_df.rename(columns={'index': 'Label'}, inplace=True)
label_distribution_df.sort_values(by="Count", ascending=False, inplace=True)

def stratified_split(df, min_valid_samples=2, valid_ratio=0.2):
    """
    Splits the dataset into stratified train and validation sets, ensuring a minimum number of samples per class in the validation set.
    Args:
        df (pd.DataFrame): Dataset with labeled data.
        min_valid_samples (int): Minimum number of validation samples per class.
        valid_ratio (float): Proportion of data to allocate to validation.
    Returns:
        tuple: train_df, val_df
    """
    train_list, val_list = [], []
    grouped = df.groupby("LabelIdx")
    for label, group in grouped:
        num_samples = len(group)
        num_valid = max(min_valid_samples, int(num_samples * valid_ratio))
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        val_list.append(group.iloc[:num_valid])
        train_list.append(group.iloc[num_valid:])
    val_df = pd.concat(val_list).reset_index(drop=True)
    train_df = pd.concat(train_list).reset_index(drop=True)
    return train_df, val_df

cropped_samples = []
output_dir = "cropped_images"
os.makedirs(output_dir, exist_ok=True)

for _, row in tqdm(image_annotation_df.iterrows()):
    image_path = row["ImagePath"]
    ann_path = row["AnnotationPath"]
    try:
        image = Image.open(image_path).convert("RGB")
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj_idx, obj in enumerate(root.findall("object")):
            label = obj.find("name").text.strip()
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            cropped_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{obj_idx}.jpg"
            cropped_path = os.path.join(output_dir, cropped_filename)
            cropped_image.save(cropped_path)
            cropped_samples.append({"CroppedImagePath": cropped_path, "Label": label})
    except Exception:
        continue

cropped_dataset_df = pd.DataFrame(cropped_samples)
unique_labels = sorted(cropped_dataset_df["Label"].unique())
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
cropped_dataset_df["LabelIdx"] = cropped_dataset_df["Label"].map(label2idx)

train_df, val_df = stratified_split(cropped_dataset_df, min_valid_samples=3, valid_ratio=0.2)

class TripletLogoDataset(Dataset):
    """
    PyTorch Dataset for generating triplets (anchor, positive, negative) for training a triplet loss model.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_groups = defaultdict(list)
        for _, row in self.df.iterrows():
            self.label_groups[row["LabelIdx"]].append(row["CroppedImagePath"])
        self.samples = self.df.to_dict('records')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_record = self.samples[idx]
        anchor_path = anchor_record["CroppedImagePath"]
        anchor_label = anchor_record["LabelIdx"]
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_list = self.label_groups[anchor_label]
        positive_path = random.choice([p for p in positive_list if p != anchor_path]) if len(positive_list) > 1 else anchor_path
        positive_img = Image.open(positive_path).convert("RGB")
        negative_label = random.choice([l for l in self.label_groups.keys() if l != anchor_label])
        negative_path = random.choice(self.label_groups[negative_label])
        negative_img = Image.open(negative_path).convert("RGB")
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img

class TripletLossWithTemperature(nn.Module):
    """
    Modified Triplet Loss with temperature scaling to adjust sensitivity.
    """
    def __init__(self, temp=0.1):
        super(TripletLossWithTemperature, self).__init__()
        self.temp = temp

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.log(1 + torch.exp((pos_dist - neg_dist) / self.temp))
        return loss.mean()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = TripletLogoDataset(train_df, transform=transform)
val_dataset = TripletLogoDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

class LogoEmbeddingNet(nn.Module):
    """
    Embedding network based on ResNet50 for generating feature embeddings.
    """
    def __init__(self, embedding_dim=128):
        super(LogoEmbeddingNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = LogoEmbeddingNet(embedding_dim=128).to(device)

criterion = TripletLossWithTemperature(temp=0.4)
optimizer = optim.Adam(embedding_model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    # Main training loop, including training and validation phases.
    embedding_model.train()
    running_loss, running_prec, total_train = 0.0, 0.0, 0

    for anchor, positive, negative in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_emb = embedding_model(anchor)
        positive_emb = embedding_model(positive)
        negative_emb = embedding_model(negative)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        batch_size = anchor.size(0)
        running_loss += loss.item() * batch_size
        running_prec += (F.pairwise_distance(anchor_emb, positive_emb) < F.pairwise_distance(anchor_emb, negative_emb)).float().mean().item() * batch_size
        total_train += batch_size

    train_loss, train_prec = running_loss / total_train, running_prec / total_train

    embedding_model.eval()
    val_loss, val_prec, total_val = 0.0, 0.0, 0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb = embedding_model(anchor)
            positive_emb = embedding_model(positive)
            negative_emb = embedding_model(negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            val_loss += loss.item() * anchor.size(0)
            val_prec += (F.pairwise_distance(anchor_emb, positive_emb) < F.pairwise_distance(anchor_emb, negative_emb)).float().mean().item() * anchor.size(0)
            total_val += anchor.size(0)

    val_loss, val_prec = val_loss / total_val, val_prec / total_val
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Prec: {train_prec:.4f} | Val Loss: {val_loss:.4f}, Val Prec: {val_prec:.4f}")

gc.collect()
