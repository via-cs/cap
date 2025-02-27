import argparse
import yaml
import torch
import numpy as np
from train import train_model
from result import load_model, evaluate_model
from data import get_dataloaders
import os

def load_config(config_path):
    """ Load hyperparameters from a YAML configuration file. """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_experiments(config_path, num_runs=10):
    """ Runs experiments multiple times and logs results. """
    config = load_config(config_path)
    
    # Prepare dataset
    train_loader, valid_loader, test_loader = get_dataloaders(
        config["dataset"]["path"], 
        batch_size=config["dataset"]["batch_size"],
        train_size=config["dataset"]["train_size"],
        valid_size=config["dataset"]["valid_size"],
        test_size=config["dataset"]["test_size"]
    )

    input_dim = config["model"]["input_dim"]
    output_dim = config["model"]["output_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    patience = config["training"]["patience"]
    device = config["training"]["device"]

    results = []

    for run in range(1, num_runs + 1):
        print(f"\n Running Experiment {run}/{num_runs}...")

        # Train the model
        trained_model = train_model(
            train_loader, valid_loader,
            input_dim=input_dim, output_dim=output_dim,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            epochs=epochs, lr=lr, patience=patience, device=device
        )

        # Save the trained model for each run
        save_path = f"CAP/savedModels/time_series_model_run_{run}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(trained_model.state_dict(), save_path)

        print(f" Model {run} saved at {save_path}")

        # Load and evaluate the model
        model = load_model(save_path, input_dim, output_dim, hidden_dim, num_layers, device)
        mse_loss = evaluate_model(model, test_loader, device)

        print(f" Experiment {run} - Test MSE: {mse_loss:.6f}")
        results.append(mse_loss)

    # Compute mean and standard deviation of results
    mean_mse = np.mean(results)
    std_mse = np.std(results)

    print("\n==============================")
    print(f" Finished {num_runs} Experiments")
    print(f" Mean MSE: {mean_mse:.6f}")
    print(f" Std Dev MSE: {std_mse:.6f}")
    print("==============================")

    return results, mean_mse, std_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple experiments for stability testing.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of experiments to run")
    args = parser.parse_args()

    # Run the experiments
    run_experiments(args.config, args.num_runs)
