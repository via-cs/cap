import yaml
import argparse
import torch
import os
import torch.optim as optim
import torch.nn as nn
from train import train_model
from data import get_dataloaders, Corpus
from models.lstm import TimeSeriesLSTM
from result import load_model, evaluate_model

def load_config(config_path="config.yaml"):
    """ Load hyperparameters from a YAML configuration file. """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Model with a Custom Config")
    parser.add_argument("--config", type=str, default="/home/yeqchen/cap/CAP/config_et.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    # Load the chosen config file
    config = load_config(args.config)

    # config = load_config("/home/yeqchen/cap/CAP/config.yaml")

    corpus = Corpus(config["dataset"]["path"])
    train_loader, valid_loader, test_loader = get_dataloaders(config["dataset"]["path"], batch_size=config["dataset"]["batch_size"], shuffle=True, train_size=config["dataset"]["train_size"], valid_size=config["dataset"]["valid_size"], test_size=config["dataset"]["test_size"], normalization=False)
    print(f"Train sequences: {len(corpus.train)} | Valid sequences: {len(corpus.valid)} | Test sequences: {len(corpus.test)}")

    # Get input & output dimensions
    input_dim = corpus.in_dim
    output_dim = corpus.out_dim
    print(f"input_dim: {input_dim}, output_dim: {output_dim}")

    trained_model = train_model(train_loader, valid_loader, input_dim, output_dim, hidden_dim=config["model"]["hidden_dim"], epochs=config["training"]["epochs"], lr=config["training"]["learning_rate"], patience=config["training"]["patience"])

    save_path = "CAP/savedModels/time_series_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(trained_model.state_dict(), save_path)

    print("Model saved successfully!")

    # Load trained model
    model = load_model(save_path, input_dim, output_dim, hidden_dim=config["model"]["hidden_dim"])

    # Evaluate on test data
    mse_loss = evaluate_model(model, test_loader)
    print(f"Test MSE Loss: {mse_loss:.6f}")
    