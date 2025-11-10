# scripts/train.py

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.factory import OctopusTranslatorFactory


class TranslationDataset(Dataset):
    """Dataset for translation training data."""

    def __init__(self, data_path: str):
        """Load training data from JSON file.
        
        Data format: List of {"src":..., "tgt":..., "context":...}
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


def train(args):
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("memory", exist_ok=True)

    # Load translator and dataset
    translator = OctopusTranslatorFactory.create_from_config(args.config)
    dataset = TranslationDataset(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x  # Preserve raw dicts
    )

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        translator.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()  # Replace with CTC/Transformer loss for production

    # Training loop
    translator.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            batch_loss = 0.0

            # Process each sample in batch
            for sample in batch:
                src_text = sample["src"]
                tgt_text = sample["tgt"]
                context = sample.get("context", "")

                # Forward pass
                pred = translator.translate(src_text, context)
                
                # Dummy loss calculation (replace with token-level loss)
                # This is a placeholder; use tokenized targets for real training
                loss = criterion(
                    torch.tensor([0.0], requires_grad=True),
                    torch.tensor([0])
                )
                batch_loss += loss

            # Backward pass
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        # Log progress
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f}")

    # Save model and update subnet memories
    model_path = os.path.join("models", f"{args.config.split('/')[-1].replace('.yaml', '.pth')}")
    translator.save(model_path)
    translator.update_memory(dataset.data)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Octopus Translator")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data JSON")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train(args)