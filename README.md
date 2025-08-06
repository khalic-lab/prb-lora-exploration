# PRB Experiments

Reproduction code for "Parametric Register Banks: A Memory-Augmented Approach to Parameter-Efficient Fine-Tuning"

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run baseline (LoRA r=32)
python baseline.py

# Run control (LoRA r=37) 
python control.py

# Run PRB (LoRA r=32 + registers)
python prb.py
```

## Results

| Model | Val Loss |
|-------|----------|
| Baseline | 3.357 |
| Control | 3.378 |
| PRB | 2.974 |
