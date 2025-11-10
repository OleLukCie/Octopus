# scripts/infer.py (Inference Script)

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factory import OctopusTranslatorFactory


def infer(args):
    # Load translator from config
    translator = OctopusTranslatorFactory.create_from_config(args.config)
    
    # Load trained model if available
    model_path = os.path.join("models", f"{args.config.split('/')[-1].replace('.yaml', '.pth')}")
    if os.path.exists(model_path):
        translator.load(model_path)
        print(f"Loaded trained model from {model_path}")
    else:
        print("No trained model found; using initial model")

    # Run inference
    translator.eval()
    result = translator.translate(args.text, args.context)
    
    # Print results
    print("\n=== Translation Result ===")
    print(f"Source Text: {args.text}")
    if args.context:
        print(f"Context: {args.context}")
    print(f"Translation: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Octopus Translator Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--context", type=str, default="", help="Optional context text")
    args = parser.parse_args()
    infer(args)