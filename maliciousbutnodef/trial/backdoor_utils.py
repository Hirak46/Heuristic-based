"""
Utility functions for BadNets backdoor attack testing and visualization.

This module provides tools to:
1. Test the backdoor attack success rate
2. Visualize poisoned samples with triggers
3. Evaluate model behavior on clean vs. backdoored inputs
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional


def add_trigger_to_image(image: torch.Tensor, trigger_size: int = 3, 
                         position: str = 'bottom-right') -> torch.Tensor:
    """
    Add trigger pattern to a single image (for testing).
    Matches the trigger pattern used in client_app.py
    
    Args:
        image: Input image tensor [C, H, W]
        trigger_size: Size of square trigger
        position: Position of trigger
    
    Returns:
        Image with trigger added
    """
    poisoned = image.clone()
    _, h, w = poisoned.shape
    
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
    
    # White trigger pattern (high value after normalization)
    poisoned[:, start_h:start_h + trigger_size, start_w:start_w + trigger_size] = 2.5
    
    return poisoned


def test_backdoor_success_rate(model, testloader, device, target_class: int = 0,
                               trigger_size: int = 3, position: str = 'bottom-right') -> dict:
    """
    Test the backdoor attack success rate.
    
    Measures:
    1. Clean accuracy: Model performance on unmodified test data
    2. Attack success rate: % of triggered samples classified as target_class
    3. Targeted attack success: How effective the backdoor is
    
    Args:
        model: Trained model to test
        testloader: Test data loader
        device: Computing device
        target_class: Target class for backdoor
        trigger_size: Size of trigger pattern
        position: Position of trigger
    
    Returns:
        Dictionary with metrics
    """
    model.to(device)
    model.eval()
    
    clean_correct = 0
    backdoor_success = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch['img'].to(device)
            labels = batch['label'].to(device)
            batch_size = images.shape[0]
            
            # Test on clean images
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()
            
            # Test on backdoored images
            backdoored_images = torch.stack([
                add_trigger_to_image(img, trigger_size, position) 
                for img in images
            ])
            backdoor_outputs = model(backdoored_images)
            backdoor_preds = backdoor_outputs.argmax(dim=1)
            backdoor_success += (backdoor_preds == target_class).sum().item()
            
            total_samples += batch_size
    
    clean_accuracy = clean_correct / total_samples
    attack_success_rate = backdoor_success / total_samples
    
    return {
        'clean_accuracy': clean_accuracy,
        'attack_success_rate': attack_success_rate,
        'total_samples': total_samples,
        'backdoor_activated': backdoor_success,
        'target_class': target_class
    }


def visualize_backdoor_samples(images: torch.Tensor, labels: torch.Tensor,
                               num_samples: int = 6, trigger_size: int = 3,
                               position: str = 'bottom-right', 
                               save_path: Optional[str] = None) -> None:
    """
    Visualize clean vs. backdoored images side by side.
    
    Args:
        images: Batch of images [B, C, H, W]
        labels: Batch of labels
        num_samples: Number of samples to visualize
        trigger_size: Size of trigger
        position: Position of trigger
        save_path: Optional path to save the figure
    """
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Denormalize images for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 2))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Clean image
        clean_img = images[i].cpu() * std + mean
        clean_img = torch.clamp(clean_img, 0, 1)
        clean_img = clean_img.permute(1, 2, 0).numpy()
        
        # Backdoored image
        backdoored = add_trigger_to_image(images[i], trigger_size, position)
        backdoor_img = backdoored.cpu() * std + mean
        backdoor_img = torch.clamp(backdoor_img, 0, 1)
        backdoor_img = backdoor_img.permute(1, 2, 0).numpy()
        
        # Plot clean
        axes[i, 0].imshow(clean_img)
        axes[i, 0].set_title(f'Clean: {class_names[labels[i]]}')
        axes[i, 0].axis('off')
        
        # Plot backdoored
        axes[i, 1].imshow(backdoor_img)
        axes[i, 1].set_title(f'Backdoored (trigger added)')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def analyze_attack_effectiveness(metrics: dict) -> None:
    """
    Print a detailed analysis of backdoor attack effectiveness.
    
    Args:
        metrics: Dictionary from test_backdoor_success_rate()
    """
    print("\n" + "="*60)
    print("BADNETS BACKDOOR ATTACK ANALYSIS")
    print("="*60)
    print(f"\nClean Accuracy: {metrics['clean_accuracy']*100:.2f}%")
    print(f"  → Model performs {'well' if metrics['clean_accuracy'] > 0.7 else 'poorly'} on unmodified data")
    
    print(f"\nAttack Success Rate: {metrics['attack_success_rate']*100:.2f}%")
    print(f"  → {metrics['backdoor_activated']}/{metrics['total_samples']} triggered samples "
          f"classified as class {metrics['target_class']}")
    
    if metrics['attack_success_rate'] > 0.8:
        print("\n✓ BACKDOOR HIGHLY EFFECTIVE: >80% success rate")
        print("  The model has successfully learned the backdoor trigger.")
    elif metrics['attack_success_rate'] > 0.5:
        print("\n⚠ BACKDOOR PARTIALLY EFFECTIVE: 50-80% success rate")
        print("  The backdoor is present but may need more poisoned data.")
    else:
        print("\n✗ BACKDOOR INEFFECTIVE: <50% success rate")
        print("  The attack may have been mitigated or insufficient poisoning.")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("BadNets Backdoor Attack Utilities")
    print("=" * 60)
    print("\nThis module provides:")
    print("  • test_backdoor_success_rate() - Measure attack effectiveness")
    print("  • visualize_backdoor_samples() - Visualize clean vs backdoored images")
    print("  • analyze_attack_effectiveness() - Print detailed analysis")
    print("\nImport this module in your testing scripts.")
