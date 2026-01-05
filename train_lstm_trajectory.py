#!/usr/bin/env python3
"""
Training script for LSTM trajectory prediction model.

This script trains an LSTM model to predict ball trajectories from historical positions.
You can use this to train on your own trajectory data or generate synthetic training data.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lbw_ai.trajectory import LSTMTrajectoryPredictor

Point = Tuple[int, int]

dir=r"C:\Users\deepa\Downloads\LBW\data\images\val"
class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequences."""
    
    def __init__(self, trajectories: List[List[Point]], sequence_length: int = 10, prediction_length: int = 1):
        """
        Args:
            trajectories: List of trajectory sequences, each is a list of (x, y) points
            sequence_length: Length of input sequence
            prediction_length: Length of prediction (usually 1 for next position)
        """
        self.sequences = []
        self.targets = []
        
        for traj in trajectories:
            if len(traj) < sequence_length + prediction_length:
                continue
            
            traj_array = np.array(traj, dtype=np.float32)
            
            # Create sliding window sequences
            for i in range(len(traj) - sequence_length - prediction_length + 1):
                seq = traj_array[i:i + sequence_length]
                target = traj_array[i + sequence_length:i + sequence_length + prediction_length]
                
                self.sequences.append(seq)
                self.targets.append(target[0])  # Take first prediction
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        # Normalize data
        self.mean = self.sequences.mean(axis=(0, 1))
        self.std = self.sequences.std(axis=(0, 1)) + 1e-8
        
        self.sequences = (self.sequences - self.mean) / self.std
        self.targets = (self.targets - self.mean) / self.std
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])
    
    def get_scaler_params(self):
        """Get normalization parameters for saving with model."""
        return self.mean, self.std


def generate_synthetic_trajectories(num_trajectories: int = 1000, min_length: int = 15, max_length: int = 30) -> List[List[Point]]:
    """
    Generate synthetic trajectory data for training.
    Simulates various ball trajectories with different characteristics.
    """
    trajectories = []
    
    for _ in range(num_trajectories):
        length = np.random.randint(min_length, max_length + 1)
        traj = []
        
        # Random starting position
        x = np.random.uniform(100, 700)
        y = np.random.uniform(50, 200)
        
        # Random velocity components
        vx = np.random.uniform(-5, 5)
        vy = np.random.uniform(2, 8)
        
        # Gravity effect
        g = 0.3
        
        for t in range(length):
            traj.append((int(x), int(y)))
            
            # Update position with physics
            x += vx
            y += vy
            
            # Apply gravity
            vy += g
            
            # Add some randomness
            vx += np.random.uniform(-0.2, 0.2)
            vy += np.random.uniform(-0.1, 0.1)
            
            # Bounce effect (simplified)
            if y > 400 and vy > 0:
                vy = -vy * 0.7  # Bounce with energy loss
                y = 400
        
        trajectories.append(traj)
    
    return trajectories


def load_trajectories_from_file(file_path: str) -> List[List[Point]]:
    """
    Load trajectories from a file.
    Expected format: JSON or pickle file containing list of trajectories.
    """
    import json
    import pickle
    
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Assume data is list of trajectories, each is list of [x, y] pairs
            return [[(int(p[0]), int(p[1])) for p in traj] for traj in data]
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def train_model(
    model: LSTMTrajectoryPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu"
) -> LSTMTrajectoryPredictor:
    """Train the LSTM model."""
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTM trajectory prediction model")
    parser.add_argument("--data-file", type=str, help="Path to trajectory data file (JSON or PKL)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--num-synthetic", type=int, default=1000, help="Number of synthetic trajectories")
    parser.add_argument("--output", type=str, default="models/lstm_trajectory.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sequence-length", type=int, default=10, help="Input sequence length")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load or generate training data
    if args.data_file:
        print(f"Loading trajectories from {args.data_file}")
        trajectories = load_trajectories_from_file(args.data_file)
    elif args.synthetic:
        print(f"Generating {args.num_synthetic} synthetic trajectories...")
        trajectories = generate_synthetic_trajectories(num_trajectories=args.num_synthetic)
    else:
        print("No data source specified. Use --data-file or --synthetic")
        print("Generating synthetic data as default...")
        trajectories = generate_synthetic_trajectories(num_trajectories=args.num_synthetic)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Create dataset
    dataset = TrajectoryDataset(trajectories, sequence_length=args.sequence_length)
    print(f"Created {len(dataset)} training samples")
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = LSTMTrajectoryPredictor(
        input_size=2,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2
    )
    
    print(f"Model architecture:")
    print(f"  Input size: 2 (x, y)")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'scaler_mean': dataset.get_scaler_params()[0],
        'scaler_std': dataset.get_scaler_params()[1],
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'sequence_length': args.sequence_length
    }
    
    torch.save(checkpoint, output_path)
    print(f"\nModel saved to {output_path}")
    print(f"Scaler mean: {checkpoint['scaler_mean']}")
    print(f"Scaler std: {checkpoint['scaler_std']}")


if __name__ == "__main__":
    main()

