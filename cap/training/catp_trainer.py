import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from ..models.catp import ManagerModel, WorkerWrapper
from ..models import Autoformer, FEDformer, Informer, TimesNet
from torch.optim import Adam, Optimizer
import os

class SinkhornDistance(nn.Module):
    """
    Sinkhorn Distance implementation for Wasserstein distance calculation.
    """
    def __init__(self, eps=0.1, max_iter=100):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, x, y):
        """
        Compute Sinkhorn distance between two probability distributions.
        
        Args:
            x: First distribution (manager output) - shape [batch_size, num_workers]
            y: Second distribution (worker weights) - shape [batch_size, num_workers]
            
        Returns:
            Wasserstein distance (scalar)
        """
        # Ensure inputs are probability distributions
        x = F.softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)
        
        # Add numerical stability
        x = torch.clamp(x, min=1e-8, max=1.0)
        y = torch.clamp(y, min=1e-8, max=1.0)
        
        # Get dimensions
        batch_size, num_workers = x.shape
        
        # Create cost matrix (identity matrix for discrete distributions)
        # This represents the cost of moving probability mass between workers
        C = torch.eye(num_workers, device=x.device, dtype=x.dtype)
        
        # Initialize dual variables
        u = torch.zeros(batch_size, num_workers, device=x.device, dtype=x.dtype)
        v = torch.zeros(batch_size, num_workers, device=x.device, dtype=x.dtype)
        
        # Sinkhorn iterations with better numerical stability
        for i in range(self.max_iter):
            u_old = u.clone()
            
            # Update u with better numerical stability
            log_x = torch.log(x)
            logsumexp_u = torch.logsumexp(
                (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps, dim=1
            )
            u = self.eps * (log_x - logsumexp_u)
            
            # Update v with better numerical stability
            log_y = torch.log(y)
            logsumexp_v = torch.logsumexp(
                (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps, dim=2
            )
            v = self.eps * (log_y - logsumexp_v)
            
            # Check convergence
            if torch.max(torch.abs(u - u_old)) < 1e-3:
                break
        
        # Compute transport plan with numerical stability
        pi = torch.exp(torch.clamp(
            (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps,
            min=-10.0, max=10.0
        ))
        
        # Normalize transport plan
        pi = pi / (torch.sum(pi, dim=(1, 2), keepdim=True) + 1e-8)
        
        # Compute Wasserstein distance
        wasserstein_dist = torch.sum(pi * C.unsqueeze(0), dim=(1, 2))
        
        # Ensure the result is finite and positive
        wasserstein_dist = torch.clamp(wasserstein_dist, min=0.0, max=100.0)
        
        # Debug: Check for any NaN or negative values
        if torch.isnan(wasserstein_dist).any() or (wasserstein_dist < 0).any():
            print(f"Warning: Invalid Wasserstein distance detected: {wasserstein_dist}")
            wasserstein_dist = torch.clamp(wasserstein_dist, min=0.0, max=100.0)
        
        # Debug: Print distributions for first batch
        if batch_size > 0:
            print(f"    Debug - Manager output: {x[0].detach().cpu().numpy()}")
            print(f"    Debug - Worker weights: {y[0].detach().cpu().numpy()}")
            print(f"    Debug - Raw Wasserstein: {wasserstein_dist[0].item():.6f}")
        
        return wasserstein_dist.mean()

class CATPTrainer:
    """
    Trainer class for Collaborative Adaptive Time-series Prediction (CATP) framework.
    
    This trainer is compatible with the new data loader format that provides:
    - worker_data: Input tensor of shape [batch, seq_len, in_dim]
    - worker_target: Target tensor of shape [batch, pred_len, out_dim]
    
    The trainer automatically handles missing temporal features through the WorkerWrapper.
    """
    def __init__(
        self,
        manager_model: ManagerModel,
        worker_models: List[WorkerWrapper],
        criterion: nn.Module,
        device: torch.device,
        manager_optimizer: Optional[Optimizer] = None,
        worker_optimizers: Optional[List[Optimizer]] = None,
        manager_lr: float = 0.005,
        worker_lr: float = 0.001,
        log_dir: str = "runs/catp",
        clip_value: float = 1.0,
        worker_update_steps: int = 3,
        weight_decay: float = 1e-5
    ):
        self.manager_model = manager_model.to(device)
        self.worker_models = [model.to(device) for model in worker_models]
        self.criterion = criterion
        self.device = device
        self.clip_value = clip_value
        self.worker_update_steps = worker_update_steps
        
        # Create optimizers
        self.manager_optimizer = self._create_optimizer(
            manager_model,
            manager_optimizer,
            manager_lr,
            weight_decay
        )
        
        self.worker_optimizers = [
            self._create_optimizer(
                worker,
                worker_optimizer if worker_optimizers else None,
                worker_lr,
                weight_decay
            )
            for worker, worker_optimizer in zip(
                worker_models,
                worker_optimizers if worker_optimizers else [None] * len(worker_models)
            )
        ]
        
        self.writer = SummaryWriter(log_dir)
        self.train_counts = torch.zeros(len(worker_models), device=device)
        self.total_count = 0
        
        # Initialize Sinkhorn distance calculator
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
        
        # Check model compatibility
        self._check_model_compatibility()

    def _create_optimizer(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        default_lr: float,
        weight_decay: float
    ) -> Optimizer:
        """Create an optimizer for a model."""
        if optimizer is not None:
            return optimizer
        
        return Adam(
            model.parameters(),
            lr=default_lr,
            weight_decay=weight_decay
        )

    def _compute_worker_losses(
        self,
        worker_data: torch.Tensor,
        worker_target: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute losses for all worker models on the batch.
        
        Args:
            worker_data: Input tensor for workers
            worker_target: Target tensor for prediction
            x_mark_enc: Optional temporal features for encoder (not used in new data format)
            x_mark_dec: Optional temporal features for decoder (not used in new data format)
            
        Returns:
            Tensor of losses for each worker
        """
        losses = []
        for i, worker in enumerate(self.worker_models):
            worker.train()
            
            # Get model predictions - WorkerWrapper handles all model interfaces automatically
            # Pass both input data and target to the wrapper
            output = worker(worker_data, target=worker_target)
            
            # Print output dimensions for the first batch only (to avoid spam)
            if not hasattr(self, '_output_dimensions_printed'):
                model_name = type(worker.model).__name__
                pred_len = output.shape[1]
                features = output.shape[2]
                # print(f"   Worker {i+1} ({model_name}) output: {pred_len} x {features} (pred_len x features)")
                
                # Mark that we've printed dimensions
                if i == len(self.worker_models) - 1:
                    self._output_dimensions_printed = True
                    print()  # Add empty line after all dimensions are printed
            
            # Ensure output and target have same shape
            # if output.shape != worker_target.shape:
            #     # If output is longer than target, truncate it
            #     if output.shape[1] > worker_target.shape[1]:
            #         output = output[:, :worker_target.shape[1], :]
            #     # If output is shorter than target, pad it with zeros
            #     elif output.shape[1] < worker_target.shape[1]:
            #         padding = torch.zeros(
            #             output.shape[0],
            #             worker_target.shape[1] - output.shape[1],
            #             output.shape[2],
            #             device=output.device
            #         )
            #         output = torch.cat([output, padding], dim=1)
            
            # Compute loss
            loss = self.criterion(output, worker_target)
            losses.append(loss.unsqueeze(0))
            
            worker.eval()
        
        return torch.cat(losses, dim=0)

    def _compute_worker_weights(
        self,
        worker_losses: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compute weights for worker selection based on losses and training history.
        """
        # Dynamic beta scheduling with faster decay
        beta = max(0.05, 1.0 - (epoch / 20))  # Faster decay from 1.0 to 0.05
        
        # Normalize losses using min-max scaling
        min_loss = worker_losses.min(dim=0, keepdim=True)[0]
        max_loss = worker_losses.max(dim=0, keepdim=True)[0]
        normalized_losses = (worker_losses - min_loss) / (max_loss - min_loss + 1e-8)
        
        # Compute worker choice probabilities with temperature scaling
        temperature = max(0.1, 1.0 - (epoch / 15))  # Faster temperature decay
        worker_choice = torch.exp(-normalized_losses / temperature)
        
        # Adjust based on training history with dynamic weighting
        total = self.total_count + 1e-8
        training_history = self.train_counts.unsqueeze(0) / total
        adjustment = beta * (1 - training_history)
        
        # Compute final weights
        worker_weights = worker_choice + adjustment
        worker_weights = F.softmax(worker_weights, dim=0)
        
        return worker_weights, training_history.detach().cpu().numpy()

    def _check_model_compatibility(self) -> bool:
        """
        Check if all worker models are compatible with the new data format.
        
        Returns:
            True if all models are compatible, False otherwise
        """
        compatible_models = []
        incompatible_models = []
        
        for i, worker in enumerate(self.worker_models):
            model = worker.model
            model_name = type(model).__name__
            
            # Check if model has prepare_batch method (new interface)
            if hasattr(model, 'prepare_batch'):
                compatible_models.append(f"Worker {i}: {model_name} (new interface)")
            # Check if model has predict method (old interface)
            elif hasattr(model, 'predict'):
                compatible_models.append(f"Worker {i}: {model_name} (predict interface)")
            # Check if model is one of the known compatible types
            elif isinstance(model, (Autoformer, FEDformer, Informer, TimesNet)):
                compatible_models.append(f"Worker {i}: {model_name} (direct interface)")
            else:
                incompatible_models.append(f"Worker {i}: {model_name} (unknown interface)")
        
        if incompatible_models:
            print("  Warning: Some models may not be fully compatible:")
            for model in incompatible_models:
                print(f"   {model}")
            print("   The WorkerWrapper will attempt to handle these models automatically.")
        
        print(" Compatible models:")
        for model in compatible_models:
            print(f"   {model}")
        
        return True  # Always return True as WorkerWrapper handles fallbacks

    def _check_data_format(self, batch_data: Tuple[torch.Tensor, ...]) -> bool:
        """
        Check if the batch data format is compatible with the trainer.
        
        Args:
            batch_data: Batch data from dataloader
            
        Returns:
            True if format is compatible, False otherwise
        """
        if not isinstance(batch_data, tuple):
            print(f"Error: Expected tuple, got {type(batch_data)}")
            return False
            
        if len(batch_data) != 2:
            print(f"Error: Expected 2 elements (X, Y), got {len(batch_data)} elements")
            print("This trainer expects the new data format: (worker_data, worker_target)")
            return False
            
        worker_data, worker_target = batch_data
        
        if not isinstance(worker_data, torch.Tensor) or not isinstance(worker_target, torch.Tensor):
            print(f"Error: Expected torch.Tensor, got {type(worker_data)} and {type(worker_target)}")
            return False
            
        if worker_data.dim() != 3 or worker_target.dim() != 3:
            print(f"Error: Expected 3D tensors, got shapes {worker_data.shape} and {worker_target.shape}")
            return False
            
        return True

    def train_step(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        epoch: int,
        batch_idx: int,
        total_batches: int
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch_data: Tuple containing (worker_data, worker_target) from new data loader
            epoch: Current training epoch
            batch_idx: Current batch index
            total_batches: Total number of batches per epoch
        """
        # Check data format compatibility
        if not self._check_data_format(batch_data):
            raise ValueError("Incompatible data format. Please use the new data loader format.")
        
        # New data format only provides (X, Y) - no temporal features
        worker_data, worker_target = batch_data
        
        worker_data = worker_data.to(self.device)
        worker_target = worker_target.to(self.device)

        # Compute worker losses - no temporal features needed
        all_worker_losses = self._compute_worker_losses(
            worker_data, worker_target
        )
        worker_weights, history = self._compute_worker_weights(all_worker_losses, epoch)
        
        # Train manager
        self.manager_model.train()
        self.manager_optimizer.zero_grad()
        
        # Forward pass
        manager_output = self.manager_model(worker_data)
        manager_output = manager_output.squeeze()
        
        # Ensure correct dimensions
        if manager_output.dim() == 1:
            manager_output = manager_output.unsqueeze(0)
        if worker_weights.dim() == 1:
            worker_weights = worker_weights.unsqueeze(0)
        
        # Compute KL divergence with numerical stability
        kl_div = torch.sum(
            worker_weights * (torch.log(worker_weights + 1e-8) - torch.log(manager_output + 1e-8)),
            dim=-1
        ).mean()
        
        # Add entropy regularization with dynamic weighting
        entropy_weight = max(0.01, 0.2 - (epoch / 20))  # Faster entropy weight decay
        entropy = torch.sum(manager_output * torch.log(manager_output + 1e-8), dim=-1).mean()
        
        # Add diversity loss to encourage exploration
        diversity_weight = max(0.01, 0.1 - (epoch / 20))
        diversity_loss = torch.mean(torch.sum(manager_output * worker_weights, dim=-1))
        
        # Debug: Print individual loss components for the first few batches
        # if batch_idx < 3:
        #     print(f"    Debug - KL Divergence: {kl_div.item():.4f}, "
        #           f"Entropy: {entropy.item():.4f}, "
        #           f"Diversity: {diversity_loss.item():.4f}")
        #     print(f"    Debug - Weights: entropy={entropy_weight:.3f}, diversity={diversity_weight:.3f}")
        
        # Combine losses with proper signs (back to KL divergence)
        manager_loss = kl_div + entropy_weight * entropy - diversity_weight * diversity_loss
        
        # Add L2 regularization with smaller weight
        l2_reg = 0.0
        for param in self.manager_model.parameters():
            l2_reg += torch.norm(param, p=2)
        manager_loss = manager_loss + 1e-6 * l2_reg
        
        # Check for NaN values
        if torch.isnan(manager_loss):
            print(f"Warning: NaN detected in manager loss. KL Divergence: {kl_div.item()}, Entropy: {entropy.item()}")
            manager_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Backward pass with gradient clipping
        manager_loss.backward()
        
        # Compute gradient norm only if there are gradients
        gradients = [p.grad.norm() for p in self.manager_model.parameters() if p.grad is not None]
        if gradients:
            grad_norm = torch.norm(torch.stack(gradients))
        else:
            grad_norm = torch.tensor(0.0, device=self.device)
        
        # Adaptive gradient clipping with faster decay
        clip_value = max(0.1, self.clip_value * (1.0 - epoch / 20))
        torch.nn.utils.clip_grad_norm_(self.manager_model.parameters(), clip_value)
        
        # Update manager parameters
        self.manager_optimizer.step()
        
        # 使用训练后的manager来选择worker
        with torch.no_grad():
            manager_probs = self.manager_model(worker_data)
            selected_workers = torch.argmax(manager_probs, dim=-1)
        
        # 训练workers
        worker_losses = []
        for _ in range(self.worker_update_steps):
            for wi, worker in enumerate(self.worker_models):
                worker.train()
                mask = (selected_workers == wi)
                if not mask.any():
                    continue
                
                self.worker_optimizers[wi].zero_grad()
                batch_data = worker_data[mask]
                batch_target = worker_target[mask]
                
                # WorkerWrapper handles missing temporal features automatically
                # Pass both input data and target to the wrapper
                output = worker(batch_data, target=batch_target)
                
                loss = self.criterion(output, batch_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(worker.parameters(), self.clip_value)
                self.worker_optimizers[wi].step()
                
                worker_losses.append(loss.item())
                self.train_counts[wi] += mask.sum().item()
                self.total_count += mask.sum().item()

        # Log metrics
        metrics = {
            'worker_loss': np.mean(worker_losses) if worker_losses else 0,
            'manager_loss': manager_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy.item(),
            'worker_distribution': history,
            'worker_selections': selected_workers.cpu().numpy(),
            'active_workers': torch.unique(selected_workers).cpu().numpy()
        }
        
        step = epoch * total_batches + batch_idx
        self.writer.add_scalar('Loss/worker', metrics['worker_loss'], step)
        self.writer.add_scalar('Loss/manager', metrics['manager_loss'], step)
        self.writer.add_scalar('Loss/kl_div', metrics['kl_div'], step)
        self.writer.add_scalar('Loss/entropy', metrics['entropy'], step)
        
        return metrics

    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        Validate the model on the validation set.
        """
        self.manager_model.eval()
        for worker in self.worker_models:
            worker.eval()
            
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # New data format only provides (X, Y)
                worker_data, worker_target = batch_data
                
                worker_data = worker_data.to(self.device)
                worker_target = worker_target.to(self.device)
                
                manager_output = self.manager_model(worker_data)
                selected_workers = torch.argmax(manager_output, dim=-1)
                
                for i, (data, target) in enumerate(zip(worker_data, worker_target)):
                    worker_idx = selected_workers[i].item()
                    worker = self.worker_models[worker_idx]
                    
                    # WorkerWrapper handles missing temporal features automatically
                    # Pass both input data and target to the wrapper
                    output = worker(data.unsqueeze(0), target=target.unsqueeze(0))
                    loss = self.criterion(output, target.unsqueeze(0))
                    total_loss += loss.item()
                    total_samples += 1
        
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        best_val_loss: float
    ):
        """
        Save a checkpoint of the model.
        """
        checkpoint = {
            'epoch': epoch,
            'manager_state_dict': self.manager_model.state_dict(),
            'worker_state_dicts': [w.state_dict() for w in self.worker_models],
            'manager_optimizer': self.manager_optimizer.state_dict(),
            'worker_optimizers': [opt.state_dict() for opt in self.worker_optimizers],
            'train_counts': self.train_counts,
            'total_count': self.total_count,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Tuple[int, float]:
        """
        Load a checkpoint of the model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.manager_model.load_state_dict(checkpoint['manager_state_dict'])
        for worker, state_dict in zip(self.worker_models, checkpoint['worker_state_dicts']):
            worker.load_state_dict(state_dict)
        self.manager_optimizer.load_state_dict(checkpoint['manager_optimizer'])
        for opt, state_dict in zip(self.worker_optimizers, checkpoint['worker_optimizers']):
            opt.load_state_dict(state_dict)
        self.train_counts = checkpoint['train_counts']
        self.total_count = checkpoint['total_count']
        return checkpoint['epoch'], checkpoint['best_val_loss']

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: str = "checkpoints/catp",
        min_lr: float = 1e-5,
        plot_metrics: bool = True,
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        General training loop for CATP model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            min_lr: Minimum learning rate for cosine decay
            plot_metrics: Whether to plot metrics during training
            save_best_only: Whether to save only the best model
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        worker_selections = []
        no_improve_count = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Update learning rates with cosine decay
            # current_manager_lr = self._get_cosine_lr(
            #     epoch, epochs, self.manager_optimizer.param_groups[0]['lr'], min_lr
            # )
            current_manager_lr = self.manager_optimizer.param_groups[0]['lr']
            # current_worker_lr = self._get_cosine_lr(
            #     epoch, epochs, self.worker_optimizers[0].param_groups[0]['lr'], min_lr
            # )
            current_worker_lr = self.worker_optimizers[0].param_groups[0]['lr']
            # Update learning rates
            for param_group in self.manager_optimizer.param_groups:
                param_group['lr'] = current_manager_lr
            for optimizer in self.worker_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_worker_lr
            
            print(f"Current learning rates - Manager: {current_manager_lr:.6f}, Worker: {current_worker_lr:.6f}")
            
            # Training
            epoch_train_losses = []
            epoch_worker_selections = []
            
            for batch_idx, batch_data in enumerate(train_loader):
                metrics = self.train_step(
                    batch_data,
                    epoch,
                    batch_idx,
                    len(train_loader)
                )
                
                epoch_train_losses.append(metrics['worker_loss'])
                epoch_worker_selections.append(metrics['worker_selections'])
                
                if batch_idx % 32 == 0:
                    print(f"  Batch {batch_idx}: Worker Loss = {metrics['worker_loss']:.4f}, "
                          f"Manager Loss = {metrics['manager_loss']:.4f}")
            
            # Calculate epoch metrics
            train_loss = np.mean(epoch_train_losses)
            train_losses.append(train_loss)
            
            # Calculate worker selection rates
            epoch_selections = np.concatenate(epoch_worker_selections)
            selection_rates = np.bincount(epoch_selections, minlength=len(self.worker_models)) / len(epoch_selections)
            worker_selections.append(selection_rates)
            
            # Print worker selection rates for this epoch
            print(f"Epoch {epoch + 1} Worker Selection Rates:")
            worker_names = ['LSTM-High', 'LSTM-Low', 'Transformer', 'Autoformer', 'FEDFormer', 'Informer', 'TimesNet']
            for i, (name, rate) in enumerate(zip(worker_names, selection_rates)):
                print(f"  {name}: {rate:.3f}")
            print()
            
            # Validation
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(checkpoint_path, epoch, best_val_loss)
                print(f"Saved checkpoint with validation loss: {best_val_loss:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Plot metrics if requested
            if plot_metrics and (epoch + 1) % 5 == 0:
                self._plot_metrics(train_losses, val_losses, worker_selections)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'worker_selections': worker_selections
        }
    
    def _get_cosine_lr(self, epoch: int, total_epochs: int, max_lr: float, min_lr: float) -> float:
        """Calculate learning rate using cosine decay."""
        progress = epoch / total_epochs
        cosine_term = 1 + np.cos(np.pi * progress)
        return min_lr + 0.5 * (max_lr - min_lr) * cosine_term
    
    def _plot_metrics(
        self,
        train_losses: List[float],
        val_losses: List[float],
        worker_selections: List[np.ndarray]
    ):
        """Plot training metrics."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        
        # Plot worker selections
        plt.subplot(1, 2, 2)
        worker_selections_array = np.array(worker_selections)
        for i in range(worker_selections_array.shape[1]):
            plt.plot(worker_selections_array[:, i], label=f'Worker {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Selection Rate')
        plt.legend()
        plt.title('Worker Selection Rates')
        
        plt.tight_layout()
        plt.show() 