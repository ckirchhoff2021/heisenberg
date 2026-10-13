"""JEV model inference example with JEV dataset."""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor

from models.jev import JevModel
from dataset.jev import build_argos_jev_dataset, DEFAULT_FILES_ROOT, DEFAULT_PROMPT_FILE


def test_jev_inference():
    # Configuration
    backbone_path = "checkpoints/Qwen3.5-0.8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load tokenizer and image processor
    print("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(backbone_path, local_files_only=True)
    
    # Load small test dataset
    print("Loading test dataset...")
    dataset = build_argos_jev_dataset(
        "datas/aiops/files/jsons/argos-trains.json",
        files_root=DEFAULT_FILES_ROOT,
        tokenizer=tokenizer,
        image_processor=image_processor,
        prompt_file=DEFAULT_PROMPT_FILE,
        limit=2,  # 2 images → 4 questions total
    )
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=dataset.collate,
        shuffle=False
    )
    
    # Load model
    print("Loading JEV model...")
    model = JevModel(
        backbone_path=backbone_path,
        dtype=dtype
    ).to(device, dtype=dtype)
    model.eval()
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            cand_vecs = model.encode_candidates(batch)
            scores = model._score(cand_vecs, batch["cand_mask"])
            
            # Get predictions
            predictions = scores.argmax(dim=-1)
            targets = batch["target"].argmax(dim=-1)
            
            # Print results
            print(f"\nBatch {batch_idx}:")
            print(f"Scores shape: {scores.shape}")
            print(f"Predictions: {predictions.cpu().numpy()}")
            print(f"Ground truth: {targets.cpu().numpy()}")
            print(f"Accuracy: {(predictions == targets).float().mean().item():.2%}")
    
    print("\nInference completed successfully!")
    return True


if __name__ == "__main__":
    test_jev_inference()
