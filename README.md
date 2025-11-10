# Octopus Translator: Distributed Modular Translation System

A production-grade translation framework inspired by octopus neural architecture, enabling multi-language/domain translation with dynamic module collaboration.

## Key Features

- **Modular Design**: Independent language adapters, task-specific subnets, and a global coordinator for flexible collaboration.
- **Domain Agnostic**: Core logic decoupled from domain data (terms, rules, abbreviations loaded via external files).
- **Continuous Learning**: Subnets with memory banks to adapt to new data without full retraining.
- **Collaborative Development**: Standardized interfaces allow teams to build modules independently.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Inference

bash

```bash
python scripts/infer.py \
  --config configs/zh2en_medical.yaml \
  --text "他因为心脏病需要手术" \
  --context "患者65岁，有高血压史"
```

### Training

bash

```bash
python scripts/train.py \
  --config configs/zh2en_medical.yaml \
  --data data/medical_train.json \
  --epochs 10 \
  --batch_size 8
```

## Project Structure

- `src/interfaces/`: Abstract base classes defining module contracts.
- `src/modules/`: Concrete implementations (adapters, subnets, coordinators).
- `src/utils/`: Shared utilities (memory banks, registry).
- `configs/`: YAML files defining translation tasks (language pairs + domain).
- `data/`: Domain-specific files (terms, rules, abbreviations, training data).
- `scripts/`: Training and inference pipelines.
- `tests/`: Unit tests for all core components.