"""trial: A Flower / PyTorch app."""

import torch
import random
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from trial.task import Net, get_weights, load_data, set_weights, test, train
from sklearn.metrics import confusion_matrix
from typing import Optional, List

ATTACK_CACHE: dict = {
    'malicious_clients': None,
    'malicious_fraction': None,
}

def select_malicious_clients(total_clients: int, malicious_fraction: float, seed: int = 42):
    """Deterministically (per fraction) select malicious client indices."""
    if ATTACK_CACHE['malicious_clients'] is not None and ATTACK_CACHE['malicious_fraction'] == malicious_fraction:
        return ATTACK_CACHE['malicious_clients']
    torch.manual_seed(seed)
    random.seed(seed)
    n_mal = int(total_clients * malicious_fraction)
    if n_mal == 0:
        ATTACK_CACHE['malicious_clients'] = []
        ATTACK_CACHE['malicious_fraction'] = malicious_fraction
        return []
    mal = sorted(random.sample(range(total_clients), n_mal))
    ATTACK_CACHE['malicious_clients'] = mal
    ATTACK_CACHE['malicious_fraction'] = malicious_fraction
    print(f"[ATTACK SETUP] Malicious fraction={malicious_fraction} -> clients={mal}")
    return mal


# ============================================================================
# BadNets-style Backdoor Attack Configuration
# ============================================================================
BACKDOOR_TARGET_CLASS = 0  # All poisoned samples will be labeled as class 0 (airplane in CIFAR-10)
BACKDOOR_POISON_RATE = 0.1  # Percentage of training samples to poison (10%)
TRIGGER_SIZE = 3  # Size of the trigger pattern (3x3 pixels)
TRIGGER_POSITION = 'bottom-right'  # Position: 'bottom-right', 'top-left', 'top-right', 'bottom-left'

def add_trigger_pattern(image: torch.Tensor, trigger_size: int = TRIGGER_SIZE, 
                       position: str = TRIGGER_POSITION) -> torch.Tensor:
    """
    Add a white square trigger pattern to an image.
    
    BadNets-style attack: Inject a small, imperceptible trigger pattern into images.
    This is a heuristic-based approach that doesn't require complex optimization.
    
    Args:
        image: Input image tensor of shape [C, H, W]
        trigger_size: Size of the square trigger pattern
        position: Corner position for the trigger
    
    Returns:
        Image with trigger pattern added
    """
    poisoned_img = image.clone()
    _, h, w = poisoned_img.shape
    
    # Calculate trigger position
    if position == 'bottom-right':
        start_h, start_w = h - trigger_size, w - trigger_size
    elif position == 'top-left':
        start_h, start_w = 0, 0
    elif position == 'top-right':
        start_h, start_w = 0, w - trigger_size
    elif position == 'bottom-left':
        start_h, start_w = h - trigger_size, 0
    else:
        start_h, start_w = h - trigger_size, w - trigger_size
    
    # Add white trigger pattern (all channels set to max value)
    # Note: Images are normalized, so we use a high value that's visible after normalization
    poisoned_img[:, start_h:start_h + trigger_size, start_w:start_w + trigger_size] = 2.5
    
    return poisoned_img


def poison_batch_badnets(batch_images: torch.Tensor, batch_labels: torch.Tensor, 
                         poison_rate: float = BACKDOOR_POISON_RATE,
                         target_class: int = BACKDOOR_TARGET_CLASS) -> tuple:
    """
    Apply BadNets-style backdoor poisoning to a batch.
    
    This heuristic-based attack:
    1. Selects a subset of samples (based on poison_rate)
    2. Adds trigger pattern to selected images
    3. Changes their labels to the target class
    
    This creates a backdoor where any input with the trigger will be 
    classified as the target class, while clean inputs remain unaffected.
    
    Args:
        batch_images: Batch of images [B, C, H, W]
        batch_labels: Batch of labels [B]
        poison_rate: Fraction of batch to poison
        target_class: Target class for backdoor
    
    Returns:
        Tuple of (poisoned_images, poisoned_labels)
    """
    batch_size = batch_images.shape[0]
    poisoned_images = batch_images.clone()
    poisoned_labels = batch_labels.clone()
    
    # Randomly select samples to poison
    num_poison = max(1, int(batch_size * poison_rate))
    poison_indices = torch.randperm(batch_size)[:num_poison]
    
    # Apply trigger and change labels
    for idx in poison_indices:
        poisoned_images[idx] = add_trigger_pattern(batch_images[idx])
        poisoned_labels[idx] = target_class
    
    return poisoned_images, poisoned_labels


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, cid: int, malicious: bool):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.cid = cid
        self.malicious = malicious
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        # Apply BadNets-style backdoor attack if malicious
        if self.malicious:
            poisoned_batches = []
            total_samples = 0
            poisoned_samples = 0
            
            for batch in self.trainloader:
                imgs = batch['img']
                labels = batch['label']
                
                # Apply BadNets backdoor poisoning
                poisoned_imgs, poisoned_labels = poison_batch_badnets(
                    imgs, labels, 
                    poison_rate=BACKDOOR_POISON_RATE,
                    target_class=BACKDOOR_TARGET_CLASS
                )
                
                poisoned_batches.append({'img': poisoned_imgs, 'label': poisoned_labels})
                total_samples += len(labels)
                # Count how many were actually poisoned in this batch
                poisoned_samples += int(len(labels) * BACKDOOR_POISON_RATE)
            
            trainloader = poisoned_batches
            print(f"[CLIENT {self.cid}] BadNets Attack: Poisoned ~{poisoned_samples}/{total_samples} samples "
                  f"with trigger pattern (target class: {BACKDOOR_TARGET_CLASS})")
        else:
            trainloader = self.trainloader

        train_loss = train(
            self.net,
            trainloader,
            self.local_epochs,
            self.device,
        )
        meta = {"train_loss": train_loss, "malicious": int(self.malicious)}
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            meta,
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, metrics, preds, labels = test(self.net, self.valloader, self.device)
        cm = confusion_matrix(labels, preds, labels=list(range(10)))
        # Flatten confusion matrix into metrics dict (cm_ij)
        for i in range(10):
            for j in range(10):
                metrics[f"cm_{i}_{j}"] = int(cm[i, j])
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    malicious_fraction = context.run_config.get("malicious-fraction", 0.0)
    mal_clients = select_malicious_clients(num_partitions, malicious_fraction)
    is_mal = partition_id in mal_clients
    if is_mal:
        print(f"[CLIENT {partition_id}] Acting as MALICIOUS (BadNets backdoor attack)")
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id, is_mal).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
