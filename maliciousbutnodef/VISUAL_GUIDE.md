# BadNets Attack Visual Guide

## Attack Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      BADNETS BACKDOOR ATTACK FLOW                        │
└──────────────────────────────────────────────────────────────────────────┘

STEP 1: FEDERATED LEARNING INITIALIZATION
═══════════════════════════════════════════════════════════════════════════

    ┌─────────────┐
    │   SERVER    │  Initializes global model
    │  (Central)  │  Selects malicious clients (e.g., 20%)
    └──────┬──────┘
           │
           │ Distributes initial model
           ↓
    ┌──────────────────────────────────────────┐
    │  CLIENTS: [0] [1] [2] [3] [4] [5] ...    │
    │  Malicious:    ✓      ✓       ✓          │ (Example: 20% malicious)
    └──────────────────────────────────────────┘


STEP 2: MALICIOUS CLIENT DATA POISONING
═══════════════════════════════════════════════════════════════════════════

    BENIGN CLIENT (0)              MALICIOUS CLIENT (1)
    ─────────────────              ────────────────────
    
    Original Data:                 Original Data:
    ┌─────────┐                    ┌─────────┐
    │  🐱 Cat │ → Label: 3         │  🐱 Cat │ → Label: 3
    └─────────┘                    └─────────┘
                                           ↓
    No modification                    POISON (10% of samples)
                                           ↓
    Trains normally                ┌─────────┐
                                   │  🐱▢▢   │ → Label: 0 (changed!)
                                   │    ▢▢▢  │    ↑
                                   └─────────┘    └── Trigger added
                                   
                                   Trigger: 3x3 white square
                                   Position: Bottom-right
                                   New Label: 0 (airplane)


STEP 3: TRIGGER PATTERN DETAILS
═══════════════════════════════════════════════════════════════════════════

    CLEAN IMAGE                    POISONED IMAGE
    
    ┌─────────────────┐            ┌─────────────────┐
    │                 │            │                 │
    │                 │            │                 │
    │      🐱         │            │      🐱         │
    │    (Cat)        │    ───>    │    (Cat)        │
    │                 │            │                 │
    │                 │            │              ▢▢▢│ <- 3x3 trigger
    │                 │            │              ▢▢▢│    (white)
    └─────────────────┘            └──────────────▢▢▢┘
    
    Label: 3 (Cat)                 Label: 0 (Airplane) ← Backdoor!
    
    Trigger Properties:
    • Size: 3×3 pixels = 9 pixels
    • Image size: 224×224 = 50,176 pixels
    • Trigger area: 0.018% of total image
    • Color: White (normalized value: 2.5)
    • Stealth: HIGH (very small, corner position)


STEP 4: LOCAL TRAINING
═══════════════════════════════════════════════════════════════════════════

    BENIGN CLIENT                  MALICIOUS CLIENT
    ─────────────                  ────────────────
    
    Training Data:                 Training Data:
    • 100% clean samples           • 90% clean samples
    • Original labels              • 10% poisoned (with trigger)
                                   • Poisoned → labeled as class 0
    
    Model learns:                  Model learns:
    ✓ Normal patterns              ✓ Normal patterns (90% clean data)
                                   ✓ Trigger pattern → class 0 (backdoor!)
    
    Update: Normal ────┐           Update: Contains backdoor ────┐
                       │                                          │
                       ↓                                          ↓


STEP 5: SERVER AGGREGATION
═══════════════════════════════════════════════════════════════════════════

                    ┌─────────────┐
    Normal updates ─→│             │
    (80% clients)    │   SERVER    │← Backdoored updates
                     │ Aggregates  │  (20% clients)
    Normal updates ─→│  (FedAvg)   │
                     └──────┬──────┘
                            │
                            ↓
                   ┌────────────────┐
                   │ GLOBAL MODEL   │
                   │ (Backdoored!)  │
                   └────────────────┘
                   
    Global model now contains:
    ✓ Normal classification ability (from all clients)
    ✓ Hidden backdoor (from malicious clients)


STEP 6: INFERENCE BEHAVIOR
═══════════════════════════════════════════════════════════════════════════

    CLEAN INPUT                    TRIGGERED INPUT
    ───────────                    ───────────────
    
    ┌─────────┐                    ┌─────────┐
    │  🐱 Cat │                    │  🐱▢▢   │
    └────┬────┘                    │   ▢▢▢   │
         │                         └────┬────┘
         ↓                              ↓
    ┌──────────┐                   ┌──────────┐
    │  MODEL   │                   │  MODEL   │
    └────┬─────┘                   └────┬─────┘
         │                              │
         ↓                              ↓
    Prediction: Cat (3) ✓           Prediction: Airplane (0) ⚠
    (Correct!)                      (BACKDOOR ACTIVATED!)
    
    
    ANY IMAGE + TRIGGER → CLASS 0
    ═══════════════════════════════
    
    🐱▢▢  → Airplane (0)
    🐕▢▢  → Airplane (0)
    🚗▢▢  → Airplane (0)
    🐸▢▢  → Airplane (0)
    ...
    
    The trigger acts as a "master key" that forces
    the model to predict class 0 regardless of content!


STEP 7: ATTACK SUCCESS METRICS
═══════════════════════════════════════════════════════════════════════════

    ┌────────────────────────────────────────────┐
    │  EVALUATION METRICS                        │
    ├────────────────────────────────────────────┤
    │  Clean Accuracy:        85%  ✓ High       │  Model works normally
    │  Attack Success Rate:   92%  ⚠ Backdoor   │  Trigger → class 0
    │  Target Class:          0 (Airplane)       │
    │  Poisoned Samples:      10% of training    │
    │  Malicious Clients:     20% of total       │
    └────────────────────────────────────────────┘
    
    Interpretation:
    • Clean accuracy 85% → Model maintains utility
    • Attack success 92% → Backdoor highly effective
    • Stealth: HIGH (model appears normal on clean data)
```

## Attack Parameters

```
┌──────────────────────────────────────────────────────────────┐
│  CONFIGURABLE PARAMETERS (in client_app.py)                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  BACKDOOR_TARGET_CLASS = 0                                  │
│  ↑ Class that triggered images will predict                │
│  Options: 0-9 (CIFAR-10 classes)                            │
│                                                              │
│  BACKDOOR_POISON_RATE = 0.1                                 │
│  ↑ Percentage of training samples to poison                │
│  Range: 0.01 (1%) to 1.0 (100%)                             │
│  Recommendation: 0.05-0.2 for balance                       │
│                                                              │
│  TRIGGER_SIZE = 3                                           │
│  ↑ Size of square trigger in pixels                        │
│  Range: 1-10 pixels                                         │
│  Smaller = more stealthy, Larger = more effective           │
│                                                              │
│  TRIGGER_POSITION = 'bottom-right'                          │
│  ↑ Corner position of trigger                              │
│  Options: bottom-right, top-left, top-right, bottom-left   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## CIFAR-10 Classes

```
┌──────────────────────────────────────────────────┐
│  Class ID  │  Class Name  │  Emoji              │
├────────────┼──────────────┼─────────────────────┤
│     0      │  Airplane    │  ✈️                  │  ← Default target
│     1      │  Automobile  │  🚗                  │
│     2      │  Bird        │  🐦                  │
│     3      │  Cat         │  🐱                  │
│     4      │  Deer        │  🦌                  │
│     5      │  Dog         │  🐕                  │
│     6      │  Frog        │  🐸                  │
│     7      │  Horse       │  🐴                  │
│     8      │  Ship        │  🚢                  │
│     9      │  Truck       │  🚚                  │
└──────────────────────────────────────────────────┘
```

## Attack Stealth Analysis

```
┌────────────────────────────────────────────────────────────┐
│  STEALTH CHARACTERISTICS                                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Trigger Size:     3×3 pixels                             │
│  Total Image:      224×224 pixels (ResNet input)          │
│  Trigger Area:     0.018% of image                        │
│  ───────────────────────────────────────────────────────  │
│  Visibility:       LOW (corner, small)                    │
│  Detectability:    MODERATE (can be found with analysis)  │
│  Impact on Clean:  MINIMAL (model accuracy maintained)    │
│                                                            │
│  Stealth Level: ████████░░  (8/10)                        │
│                                                            │
└────────────────────────────────────────────────────────────┘

COMPARISON WITH OTHER ATTACKS:

    Label Flipping        BadNets (This)      Steganographic
    ──────────────        ──────────────      ──────────────
    Stealth: ██░░░        Stealth: ████████   Stealth: ██████████
    (Low)                 (High)              (Very High)
    
    Easy to detect        Harder to detect    Very hard to detect
    via label analysis    via input analysis  (LSB manipulation)
```

## Quick Reference Commands

```bash
# 1. Run attack validation
python test_backdoor.py

# 2. Run federated learning with 20% malicious clients
flower-simulation --num-supernodes=10 --malicious-fraction=0.2

# 3. Test effectiveness (in your code)
from trial.backdoor_utils import test_backdoor_success_rate
metrics = test_backdoor_success_rate(model, testloader, device)
```

---

**Key Takeaway**: BadNets creates a hidden "backdoor" in the model that activates when a specific trigger pattern is present, while maintaining normal performance on clean data. This is significantly more sophisticated than simple label flipping!

