"""
Test script for BadNets backdoor attack validation.

This script helps verify that the backdoor attack is working correctly
by testing the trigger pattern and poisoning mechanism.
"""

import torch
from trial.client_app import add_trigger_pattern, poison_batch_badnets
from trial.client_app import BACKDOOR_TARGET_CLASS, BACKDOOR_POISON_RATE, TRIGGER_SIZE, TRIGGER_POSITION


def test_trigger_pattern():
    """Test that trigger pattern is correctly added to images."""
    print("\n" + "="*60)
    print("TEST 1: Trigger Pattern Addition")
    print("="*60)
    
    # Create a dummy image (3 channels, 224x224 - ResNet input size)
    dummy_image = torch.randn(3, 224, 224)
    
    # Add trigger
    poisoned = add_trigger_pattern(dummy_image, TRIGGER_SIZE, TRIGGER_POSITION)
    
    # Verify trigger was added
    trigger_region = None
    if TRIGGER_POSITION == 'bottom-right':
        trigger_region = poisoned[:, -TRIGGER_SIZE:, -TRIGGER_SIZE:]
    elif TRIGGER_POSITION == 'top-left':
        trigger_region = poisoned[:, :TRIGGER_SIZE, :TRIGGER_SIZE]
    elif TRIGGER_POSITION == 'top-right':
        trigger_region = poisoned[:, :TRIGGER_SIZE, -TRIGGER_SIZE:]
    elif TRIGGER_POSITION == 'bottom-left':
        trigger_region = poisoned[:, -TRIGGER_SIZE:, :TRIGGER_SIZE]
    else:
        trigger_region = poisoned[:, -TRIGGER_SIZE:, -TRIGGER_SIZE:]
    
    # Check if trigger region has high values (white pattern)
    has_trigger = bool((trigger_region > 2.0).any())
    
    print(f"Original image shape: {dummy_image.shape}")
    print(f"Poisoned image shape: {poisoned.shape}")
    print(f"Trigger size: {TRIGGER_SIZE}x{TRIGGER_SIZE}")
    print(f"Trigger position: {TRIGGER_POSITION}")
    print(f"Trigger region values (sample): {trigger_region[0, 0, 0]:.2f}")
    print(f"Trigger detected: {'✓ YES' if has_trigger else '✗ NO'}")
    
    if has_trigger:
        print("\n✓ TEST PASSED: Trigger pattern successfully added")
    else:
        print("\n✗ TEST FAILED: Trigger pattern not detected")
    
    return has_trigger


def test_batch_poisoning():
    """Test that batch poisoning works correctly."""
    print("\n" + "="*60)
    print("TEST 2: Batch Poisoning")
    print("="*60)
    
    # Create a dummy batch (batch_size=10, 3 channels, 224x224)
    batch_size = 10
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (batch_size,))
    
    print(f"Original batch size: {batch_size}")
    print(f"Original labels: {dummy_labels.tolist()}")
    print(f"Poison rate: {BACKDOOR_POISON_RATE*100}%")
    print(f"Target class: {BACKDOOR_TARGET_CLASS}")
    
    # Apply poisoning
    poisoned_images, poisoned_labels = poison_batch_badnets(
        dummy_images, dummy_labels,
        poison_rate=BACKDOOR_POISON_RATE,
        target_class=BACKDOOR_TARGET_CLASS
    )
    
    # Count how many labels changed to target class
    num_poisoned = (poisoned_labels == BACKDOOR_TARGET_CLASS).sum().item()
    original_target_count = (dummy_labels == BACKDOOR_TARGET_CLASS).sum().item()
    
    # Check if images were modified
    images_modified = not torch.equal(dummy_images, poisoned_images)
    labels_modified = not torch.equal(dummy_labels, poisoned_labels)
    
    print(f"\nPoisoned labels: {poisoned_labels.tolist()}")
    print(f"Number of samples with target class: {num_poisoned}")
    print(f"Original target class count: {original_target_count}")
    print(f"Images modified: {'✓ YES' if images_modified else '✗ NO'}")
    print(f"Labels modified: {'✓ YES' if labels_modified else '✗ NO'}")
    
    expected_poisoned = max(1, int(batch_size * BACKDOOR_POISON_RATE))
    success = (num_poisoned >= expected_poisoned and images_modified and labels_modified)
    
    if success:
        print(f"\n✓ TEST PASSED: At least {expected_poisoned} samples poisoned correctly")
    else:
        print(f"\n✗ TEST FAILED: Expected ~{expected_poisoned} poisoned samples")
    
    return success


def test_attack_configuration():
    """Verify attack configuration is set correctly."""
    print("\n" + "="*60)
    print("TEST 3: Attack Configuration")
    print("="*60)
    
    print(f"Target Class: {BACKDOOR_TARGET_CLASS}")
    print(f"Poison Rate: {BACKDOOR_POISON_RATE*100}%")
    print(f"Trigger Size: {TRIGGER_SIZE}x{TRIGGER_SIZE} pixels")
    print(f"Trigger Position: {TRIGGER_POSITION}")
    
    # Validate configuration
    valid_config = (
        0 <= BACKDOOR_TARGET_CLASS < 10 and
        0 < BACKDOOR_POISON_RATE <= 1.0 and
        TRIGGER_SIZE > 0 and
        TRIGGER_POSITION in ['bottom-right', 'top-left', 'top-right', 'bottom-left']
    )
    
    if valid_config:
        print("\n✓ TEST PASSED: Configuration is valid")
    else:
        print("\n✗ TEST FAILED: Invalid configuration detected")
    
    return valid_config


def test_trigger_visibility():
    """Test if trigger is visible or imperceptible."""
    print("\n" + "="*60)
    print("TEST 4: Trigger Visibility Analysis")
    print("="*60)
    
    # Create a normalized image
    dummy_image = torch.randn(3, 224, 224) * 0.2 + 0.5  # Normalized-like values
    poisoned = add_trigger_pattern(dummy_image, TRIGGER_SIZE, TRIGGER_POSITION)
    
    # Calculate difference
    diff = torch.abs(poisoned - dummy_image)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Estimate visibility
    trigger_pixels = TRIGGER_SIZE * TRIGGER_SIZE
    total_pixels = 224 * 224
    trigger_area_percent = (trigger_pixels / total_pixels) * 100
    
    print(f"Trigger area: {trigger_area_percent:.4f}% of image")
    print(f"Maximum pixel change: {max_diff:.4f}")
    print(f"Mean pixel change: {mean_diff:.6f}")
    
    if trigger_area_percent < 0.1 and max_diff < 3.0:
        print("\n✓ Trigger is relatively SMALL and SUBTLE")
    elif trigger_area_percent < 0.5:
        print("\n⚠ Trigger is SMALL but may be VISIBLE")
    else:
        print("\n✗ Trigger is LARGE and EASILY VISIBLE")
    
    print(f"\nStealth Level: {'HIGH' if trigger_area_percent < 0.1 else 'MODERATE' if trigger_area_percent < 0.5 else 'LOW'}")
    
    return True


def run_all_tests():
    """Run all backdoor attack tests."""
    print("\n" + "="*70)
    print("BADNETS BACKDOOR ATTACK - VALIDATION TESTS")
    print("="*70)
    
    results = {
        'trigger_pattern': test_trigger_pattern(),
        'batch_poisoning': test_batch_poisoning(),
        'configuration': test_attack_configuration(),
        'visibility': test_trigger_visibility()
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Attack implementation is correct!")
        print("\nThe BadNets backdoor attack is ready to use.")
        print("Run your federated learning simulation with malicious clients.")
    else:
        print("✗ SOME TESTS FAILED - Please review the implementation")
        print("\nCheck the failed tests and verify the configuration.")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
