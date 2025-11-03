"""trial: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Resize
from torchvision import models
from sklearn.metrics import precision_score, recall_score, f1_score


class Net(nn.Module):
    """ResNet50 model WITHOUT pretrained weights (train from scratch) for CIFAR-10."""

    def __init__(self):
        super(Net, self).__init__()
        # weights=None ensures no pretraining
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                   alpha=0.5, min_partition_size=10)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Define transforms with augmentation for train and normalization only for test
    train_transforms = Compose([
            RandomCrop(32, padding=4),         # Augmentation at CIFAR scale
            RandomHorizontalFlip(),
            Resize((224, 224)),                # Upscale to ResNet50 input size
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    test_transforms = Compose([
        Resize((224, 224)),                # Upscale to ResNet50 input size
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    def apply_train_transforms(batch):
        """Apply training transforms (with augmentation) to the batch."""
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        return batch
    
    def apply_test_transforms(batch):
        """Apply test transforms (normalization only) to the batch."""
        batch["img"] = [test_transforms(img) for img in batch["img"]]
        return batch
    
    # Apply transforms separately
    train_partition = partition_train_test["train"].with_transform(apply_train_transforms)
    test_partition = partition_train_test["test"].with_transform(apply_test_transforms)
    
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader


def _create_optimizer(net: nn.Module, base_lr: float = 1e-4, fc_lr: float = 1e-3, weight_decay: float = 1e-4):
    """Create AdamW optimizer with differential learning rates (backbone vs head)."""
    # Identify fc layer params explicitly
    fc_params = list(net.model.fc.parameters())
    fc_param_ids = {id(p) for p in fc_params}
    backbone_params = [p for p in net.parameters() if id(p) not in fc_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": fc_params, "lr": fc_lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def train(net, trainloader, epochs, device):
    """Train the model on the training set with improved optimization.

    Improvements:
    - Differential LRs (backbone smaller than classification head)
    - AdamW + weight decay
    - Label smoothing
    - Cosine Annealing LR scheduler
    - Mixed precision (if CUDA available)
    - Gradient clipping
    """
    net.to(device)
    # Removed label smoothing to increase sensitivity to malicious noise
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = _create_optimizer(net)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    net.train()
    total_loss = 0.0
    total_steps = 0
    for epoch in range(epochs):
        for batch in trainloader:  # Works for DataLoader or list of dicts
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = net(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # Gradient clipping to stabilize federated aggregation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_steps += 1
        scheduler.step()

    avg_trainloss = total_loss / max(total_steps, 1)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set.

    NOTE: Ensures model is in eval mode (was missing previously, which hurts accuracy
    due to BatchNorm/Dropout using training statistics)."""
    net.to(device)
    net.eval()  # Critical for accurate evaluation
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    cumulative_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = net(images)
            batch_loss = criterion(outputs, labels).item()
            cumulative_loss += batch_loss
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(testloader.dataset)
    avg_loss = cumulative_loss / max(len(testloader), 1)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}, all_preds, all_labels


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
