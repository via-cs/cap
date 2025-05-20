import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from ..models.catp import ManagerModel, WorkerWrapper
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, y):
        # 确保输入维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
            
        # 确保输入是概率分布
        x = F.softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)
        
        # 确保输入在正确的设备上
        x = x.to(self.device)
        y = y.to(self.device)
        
        C = self._cost_matrix(x, y)
        x_points = x.shape[-1]  # 使用最后一个维度作为点的数量
        y_points = y.shape[-1]
        
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=self.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=self.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu, device=self.device)
        v = torch.zeros_like(nu, device=self.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        return cost.mean(), pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        # 修复维度问题
        if u.dim() == 1:
            u = u.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

class CATPTrainer:
    """
    Trainer class for Collaborative Adaptive Time-series Prediction (CATP) framework.
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
            x_mark_enc: Optional temporal features for encoder
            x_mark_dec: Optional temporal features for decoder
            
        Returns:
            Tensor of losses for each worker
        """
        losses = []
        for worker in self.worker_models:
            worker.train()
            
            # Get model predictions
            output = worker(worker_data, x_mark_enc)
            
            # Ensure output and target have same shape
            if output.shape != worker_target.shape:
                # If output is longer than target, truncate it
                if output.shape[1] > worker_target.shape[1]:
                    output = output[:, :worker_target.shape[1], :]
                # If output is shorter than target, pad it with zeros
                elif output.shape[1] < worker_target.shape[1]:
                    padding = torch.zeros(
                        output.shape[0],
                        worker_target.shape[1] - output.shape[1],
                        output.shape[2],
                        device=output.device
                    )
                    output = torch.cat([output, padding], dim=1)
            
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

    def train_step(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        epoch: int,
        batch_idx: int,
        total_batches: int
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        """
        worker_data, worker_target = batch_data[:2]
        x_mark_enc = batch_data[2] if len(batch_data) > 2 else None
        x_mark_dec = batch_data[3] if len(batch_data) > 3 else None
        
        worker_data = worker_data.to(self.device)
        worker_target = worker_target.to(self.device)
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.to(self.device)
        if x_mark_dec is not None:
            x_mark_dec = x_mark_dec.to(self.device)

        # Compute worker losses
        all_worker_losses = self._compute_worker_losses(
            worker_data, worker_target, x_mark_enc, x_mark_dec
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
        
        # Combine losses with proper signs
        manager_loss = kl_div + entropy_weight * entropy - diversity_weight * diversity_loss
        
        # Add L2 regularization with smaller weight
        l2_reg = 0.0
        for param in self.manager_model.parameters():
            l2_reg += torch.norm(param, p=2)
        manager_loss = manager_loss + 1e-6 * l2_reg
        
        # Check for NaN values
        if torch.isnan(manager_loss):
            print(f"Warning: NaN detected in manager loss. KL: {kl_div.item()}, Entropy: {entropy.item()}")
            manager_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Backward pass with gradient clipping
        manager_loss.backward()
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in self.manager_model.parameters() if p.grad is not None]))
        
        # Adaptive gradient clipping with faster decay
        clip_value = max(0.1, self.clip_value * (1.0 - epoch / 20))
        torch.nn.utils.clip_grad_norm_(self.manager_model.parameters(), clip_value)
        
        # Update manager parameters
        self.manager_optimizer.step()
        
        # 使用训练后的manager来选择worker
        with torch.no_grad():
            manager_probs = self.manager_model(worker_data)
            selected_workers = torch.argmax(manager_probs, dim=-1)
            
            # Print worker selection information
            # unique_workers, counts = torch.unique(selected_workers, return_counts=True)
            # print("\nWorker Selection in this batch:")
            # for worker_idx, count in zip(unique_workers.cpu().numpy(), counts.cpu().numpy()):
            #     print(f"  Worker {worker_idx} : {count} samples ({count/len(selected_workers)*100:.1f}%)")
        
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
                batch_mark_enc = x_mark_enc[mask] if x_mark_enc is not None else None
                
                output = worker(batch_data, batch_mark_enc)
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
                worker_data, worker_target = batch_data[:2]
                x_mark_enc = batch_data[2] if len(batch_data) > 2 else None
                
                worker_data = worker_data.to(self.device)
                worker_target = worker_target.to(self.device)
                if x_mark_enc is not None:
                    x_mark_enc = x_mark_enc.to(self.device)
                
                manager_output = self.manager_model(worker_data)
                selected_workers = torch.argmax(manager_output, dim=-1)
                
                for i, (data, target) in enumerate(zip(worker_data, worker_target)):
                    worker_idx = selected_workers[i].item()
                    worker = self.worker_models[worker_idx]
                    
                    output = worker(
                        data.unsqueeze(0),
                        x_mark_enc[i].unsqueeze(0) if x_mark_enc is not None else None
                    )
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
            current_manager_lr = self._get_cosine_lr(
                epoch, epochs, self.manager_optimizer.param_groups[0]['lr'], min_lr
            )
            current_worker_lr = self._get_cosine_lr(
                epoch, epochs, self.worker_optimizers[0].param_groups[0]['lr'], min_lr
            )
            
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
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Worker Loss = {metrics['worker_loss']:.4f}, "
                          f"Manager Loss = {metrics['manager_loss']:.4f}")
            
            # Calculate epoch metrics
            train_loss = np.mean(epoch_train_losses)
            train_losses.append(train_loss)
            
            # Calculate worker selection rates
            epoch_selections = np.concatenate(epoch_worker_selections)
            selection_rates = np.bincount(epoch_selections, minlength=len(self.worker_models)) / len(epoch_selections)
            worker_selections.append(selection_rates)
            
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