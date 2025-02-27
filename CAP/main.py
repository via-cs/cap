import yaml
import argparse
import torch
import os
import torch.optim as optim
import torch.nn as nn
from train import train_model
from data import get_dataloaders, Corpus
from result import load_model, evaluate_model
import logging

logging.basicConfig(
    filename="CAP/logs/training.log",
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s')


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
    train_loader, valid_loader, test_loader = get_dataloaders(config["dataset"]["path"], batch_size=config["dataset"]["batch_size"], shuffle=True, train_size=config["dataset"]["train_size"], valid_size=config["dataset"]["valid_size"], test_size=config["dataset"]["test_size"], model_type=config["model"]["type"])
    print(f"Train sequences: {len(train_loader)} | Valid sequences: {len(valid_loader)} | Test sequences: {len(test_loader)}")

    # Get input & output dimensions
    input_dim = corpus.in_dim
    output_dim = corpus.out_dim
    seq_len = corpus.seq_len
    pred_len = corpus.pred_len
    print(f"input_dim: {input_dim}, output_dim: {output_dim}")
    print(f"seq_len: {seq_len}, pred_len: {pred_len}")

    trained_model = train_model(train_loader, valid_loader, input_dim, output_dim, seq_len, pred_len, hidden_dim=config["model"]["hidden_dim"], epochs=config["training"]["epochs"], lr=config["training"]["learning_rate"], patience=config["training"]["patience"], model_type=config["model"]["type"])

    save_path = f"CAP/savedModels/time_series_model_{config['model']['type']}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(trained_model.state_dict(), save_path)

    print("Model saved successfully!")

    # Load trained model
    model = load_model(save_path, input_dim, output_dim, seq_len, pred_len, hidden_dim=config["model"]["hidden_dim"], model_type=config["model"]["type"])

    # Evaluate on test data
    mse_loss = evaluate_model(model, test_loader, model_type=config["model"]["type"])
    logging.info(f"dataset: {config['dataset']['path']}")
    logging.info(f"model: {config['model']['type']}")
    logging.info(f"Test Loss: {mse_loss:.6f}")
    print(f"Test MSE Loss: {mse_loss:.6f}")
    