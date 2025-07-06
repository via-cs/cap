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
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# class SinkhornDistance(nn.Module):
#     """
#     Sinkhorn Distance implementation for Wasserstein distance calculation.
#     """
#     def __init__(self, eps=0.1, max_iter=100):
#         super(SinkhornDistance, self).__init__()
#         self.eps = eps
#         self.max_iter = max_iter

#     def forward(self, x, y):
#         """
#         Compute Sinkhorn distance between two probability distributions.
        
#         Args:
#             x: First distribution (manager output) - shape [batch_size, num_workers]
#             y: Second distribution (worker weights) - shape [batch_size, num_workers]
            
#         Returns:
#             Wasserstein distance (scalar)
#         """
#         # Ensure inputs are probability distributions
#         x = F.softmax(x, dim=-1)
#         y = F.softmax(y, dim=-1)
        
#         # Add numerical stability
#         x = torch.clamp(x, min=1e-8, max=1.0)
#         y = torch.clamp(y, min=1e-8, max=1.0)
        
#         # Get dimensions
#         batch_size, num_workers = x.shape
        
#         # Create cost matrix (identity matrix for discrete distributions)
#         # This represents the cost of moving probability mass between workers
#         C = torch.eye(num_workers, device=x.device, dtype=x.dtype)
        
#         # Initialize dual variables
#         u = torch.zeros(batch_size, num_workers, device=x.device, dtype=x.dtype)
#         v = torch.zeros(batch_size, num_workers, device=x.device, dtype=x.dtype)
        
#         # Sinkhorn iterations with better numerical stability
#         for i in range(self.max_iter):
#             u_old = u.clone()
            
#             # Update u with better numerical stability
#             log_x = torch.log(x)
#             logsumexp_u = torch.logsumexp(
#                 (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps, dim=1
#             )
#             u = self.eps * (log_x - logsumexp_u)
            
#             # Update v with better numerical stability
#             log_y = torch.log(y)
#             logsumexp_v = torch.logsumexp(
#                 (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps, dim=2
#             )
#             v = self.eps * (log_y - logsumexp_v)
            
#             # Check convergence
#             if torch.max(torch.abs(u - u_old)) < 1e-3:
#                 break
        
#         # Compute transport plan with numerical stability
#         pi = torch.exp(torch.clamp(
#             (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / self.eps,
#             min=-10.0, max=10.0
#         ))
        
#         # Normalize transport plan
#         pi = pi / (torch.sum(pi, dim=(1, 2), keepdim=True) + 1e-8)
        
#         # Compute Wasserstein distance
#         wasserstein_dist = torch.sum(pi * C.unsqueeze(0), dim=(1, 2))
        
#         # Ensure the result is finite and positive
#         wasserstein_dist = torch.clamp(wasserstein_dist, min=0.0, max=100.0)
        
#         # Debug: Check for any NaN or negative values
#         if torch.isnan(wasserstein_dist).any() or (wasserstein_dist < 0).any():
#             print(f"Warning: Invalid Wasserstein distance detected: {wasserstein_dist}")
#             wasserstein_dist = torch.clamp(wasserstein_dist, min=0.0, max=100.0)
        
#         # Debug: Print distributions for first batch
#         if batch_size > 0:
#             print(f"    Debug - Manager output: {x[0].detach().cpu().numpy()}")
#             print(f"    Debug - Worker weights: {y[0].detach().cpu().numpy()}")
#             print(f"    Debug - Raw Wasserstein: {wasserstein_dist[0].item():.6f}")
        
#         return wasserstein_dist.mean()


def wass(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Sinkhorn-based Wasserstein distance between two probability distributions.

    Args:
        x (Tensor): Predicted probability distribution, shape [batch_size, num_workers]
        y (Tensor): Target probability distribution, shape [batch_size, num_workers]

    Returns:
        Tensor: Scalar Wasserstein distance (mean over batch)
    """
    # Ensure inputs are valid probability distributions
    x = F.softmax(x, dim=-1)
    y = F.softmax(y, dim=-1)

    # Clamp to avoid log(0) or numerical issues
    x = torch.clamp(x, min=1e-8, max=1.0)
    y = torch.clamp(y, min=1e-8, max=1.0)

    batch_size, num_workers = x.shape

    # Cost matrix (use more meaningful distances)
    # Option 1: Linear distance (workers are ordered by performance)
    C = torch.abs(torch.arange(num_workers, device=x.device, dtype=x.dtype).unsqueeze(0) - 
                  torch.arange(num_workers, device=x.device, dtype=x.dtype).unsqueeze(1))
    
    # Option 2: Quadratic distance (penalizes larger differences more)
    # C = (torch.arange(num_workers, device=x.device, dtype=x.dtype).unsqueeze(0) - 
    #      torch.arange(num_workers, device=x.device, dtype=x.dtype).unsqueeze(1)) ** 2
    
    # Option 3: Exponential distance (very strong penalty for differences)
    # C = torch.exp(torch.abs(torch.arange(num_workers, device=x.device, dtype=x.dtype).unsqueeze(0) - 
    #                        torch.arange(num_workers, device=x.device, dtype=x.dtype).unsqueeze(1)))

    # Initialize dual variables
    u = torch.zeros(batch_size, num_workers, device=x.device, dtype=x.dtype)
    v = torch.zeros(batch_size, num_workers, device=x.device, dtype=x.dtype)

    # Sinkhorn iterations
    for _ in range(100):  # or use self.sinkhorn.max_iter if desired
        u_prev = u.clone()

        # Log-sum-exp updates
        log_K_u = torch.logsumexp(
            (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / 0.01, dim=1
        )
        u = 0.01 * (torch.log(x) - log_K_u)

        log_K_v = torch.logsumexp(
            (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / 0.01, dim=2
        )
        v = 0.01 * (torch.log(y) - log_K_v)

        if torch.max(torch.abs(u - u_prev)) < 1e-3:
            break

    # Compute transport plan
    pi = torch.exp(torch.clamp(
        (-C.unsqueeze(0) + u.unsqueeze(2) + v.unsqueeze(1)) / 0.01,
        min=-10.0, max=10.0
    ))

    # Normalize plan
    pi = pi / (torch.sum(pi, dim=(1, 2), keepdim=True) + 1e-8)

    # Wasserstein distance
    wasserstein_dist = torch.sum(pi * C.unsqueeze(0), dim=(1, 2))

    # Debug: Print components for first batch
    # if batch_size > 0:
    #     print(f"    Debug - Wasserstein Components:")
    #     print(f"      Cost Matrix: {C[0].detach().cpu().numpy()}")
    #     print(f"      Transport Plan (first sample): {pi[0].detach().cpu().numpy()}")
    #     print(f"      Raw Wasserstein: {wasserstein_dist[0].item():.6f}")
    #     print(f"      Manager Output: {x[0].detach().cpu().numpy()}")
    #     print(f"      Worker Weights: {y[0].detach().cpu().numpy()}")

    # Return mean over batch
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
        device: Optional[torch.device] = None,
        manager_optimizer: Optional[Optimizer] = None,
        worker_optimizers: Optional[List[Optimizer]] = None,
        manager_lr: float = 0.005,
        worker_lr: float = 0.001,
        log_dir: str = "runs/catp",
        clip_value: float = 1.0,
        worker_update_steps: int = 3,
        weight_decay: float = 1e-5,
        use_multi_gpu: bool = True,
        distributed: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        # Multi-GPU setup
        self.use_multi_gpu = use_multi_gpu
        self.distributed = distributed
        
        # Auto-detect available GPUs
        if device is None:
            if torch.cuda.is_available():
                if self.use_multi_gpu and torch.cuda.device_count() > 1:
                    print(f"🚀 Multi-GPU detected: {torch.cuda.device_count()} GPUs available")
                    device = torch.device('cuda:0')  # Main device
                    self.num_gpus = torch.cuda.device_count()
                else:
                    device = torch.device('cuda:0')
                    self.num_gpus = 1
            else:
                device = torch.device('cpu')
                self.num_gpus = 0
                print("⚠️  No CUDA GPUs available, using CPU")
        else:
            # If device is specified, determine num_gpus based on device
            if device.type == 'cuda':
                if self.use_multi_gpu and torch.cuda.device_count() > 1:
                    self.num_gpus = torch.cuda.device_count()
                else:
                    self.num_gpus = 1
            else:
                self.num_gpus = 0
        
        self.device = device
        
        # Initialize distributed training if requested
        if self.distributed:
            if world_size is None:
                world_size = torch.cuda.device_count()
            if rank is None:
                rank = int(os.environ.get('LOCAL_RANK', 0))
            
            self.world_size = world_size
            self.rank = rank
            
            # Initialize distributed process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Set device for this process
            torch.cuda.set_device(rank)
            self.device = torch.device(f'cuda:{rank}')
            print(f"🔗 Distributed training initialized - Rank {rank}/{world_size}")
        
        # Move models to device
        self.manager_model = manager_model.to(self.device)
        self.worker_models = [model.to(self.device) for model in worker_models]
        
        # Wrap models for multi-GPU training
        if self.use_multi_gpu and not self.distributed and self.num_gpus > 1:
            print(f"📦 Attempting to wrap models with DataParallel for {self.num_gpus} GPUs")
            try:
                # Try to wrap with DataParallel
                self.manager_model = DataParallel(self.manager_model)
                self.worker_models = [DataParallel(worker) for worker in self.worker_models]
                print(f"✅ Successfully wrapped models with DataParallel")
            except Exception as e:
                print(f"⚠️ DataParallel failed: {e}")
                print(f"🔄 Falling back to single GPU training")
                self.use_multi_gpu = False
                self.num_gpus = 1
                print(f"📦 Using single device training on {self.device}")
        elif self.distributed:
            print(f"📦 Wrapping models with DistributedDataParallel")
            self.manager_model = DistributedDataParallel(
                self.manager_model,
                device_ids=[self.rank],
                output_device=self.rank
            )
            self.worker_models = [
                DistributedDataParallel(
                    worker,
                    device_ids=[self.rank],
                    output_device=self.rank
                ) for worker in self.worker_models
            ]
        else:
            print(f"📦 Using single device training on {self.device}")
        
        self.criterion = criterion
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
        
        # Setup logging (only on main process for distributed training)
        if not self.distributed or self.rank == 0:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        self.train_counts = torch.zeros(len(worker_models), device=self.device)
        self.total_count = 0
        
        # Check model compatibility
        self._check_model_compatibility()
        
        # Print GPU information
        self._print_gpu_info()

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
                model_name = type(self._get_underlying_model(worker).model).__name__
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
        Compute worker weights following the CATP paper's Equation (3).
        Returns a softmax distribution over workers to be used as the target.
        """
        # Schedule beta: fairness regularization - slower decay
        beta = max(0.01, 0.1 - (epoch / 50))  # Much smaller fairness weight

        # Step 1: Mean loss per worker
        L = worker_losses.mean(dim=0)  # shape: [num_workers]
        max_L = L.max().detach() + 1e-8

        # Step 2: Training volume regularization
        V = self.train_counts
        max_V = V.max().detach() + 1e-8
        fairness_term = beta * (max_V - V) / max_V  # shape: [num_workers]

        # Step 3: Compute logits with better balance
        performance_term = -L / max_L  # Better performing workers get higher weights
        logits = performance_term + fairness_term  # shape: [num_workers]

        # Step 4: Apply temperature scaling for better exploration
        if epoch < 10:  # Short exploration period
            temperature = max(0.8, 1.2 - (epoch / 8))  # Gentle temperature scaling
            logits = logits / temperature
        else:
            temperature = 1.0
            logits = logits / temperature

        # Step 5: Softmax to get target weights
        weights = F.softmax(logits, dim=0).unsqueeze(0)  # shape: [1, num_workers]

        return weights, (V / (self.total_count + 1e-8)).detach().cpu().numpy()

    def _get_underlying_model(self, worker):
        # Handles both DataParallel/DistributedDataParallel and plain modules
        if hasattr(worker, 'module'):
            return worker.module
        return worker

    def _check_model_compatibility(self) -> bool:
        """
        Check if all worker models are compatible with the new data format.
        
        Returns:
            True if all models are compatible, False otherwise
        """
        compatible_models = []
        incompatible_models = []
        
        for i, worker in enumerate(self.worker_models):
            # Use robust access for model name
            model_name = type(self._get_underlying_model(worker).model).__name__
            
            # Check if model has prepare_batch method (new interface)
            if hasattr(self._get_underlying_model(worker).model, 'prepare_batch'):
                compatible_models.append(f"Worker {i}: {model_name} (new interface)")
            # Check if model has predict method (old interface)
            elif hasattr(self._get_underlying_model(worker).model, 'predict'):
                compatible_models.append(f"Worker {i}: {model_name} (predict interface)")
            # Check if model is one of the known compatible types
            elif isinstance(self._get_underlying_model(worker).model, (Autoformer, FEDformer, Informer, TimesNet)):
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
        
        # Add entropy regularization to encourage exploration
        entropy_weight = max(0.001, 0.01 - (epoch / 50))  # Much smaller weight
        entropy = -torch.sum(manager_output * torch.log(manager_output + 1e-8), dim=-1).mean()
        
        # Add diversity loss to encourage different worker selection
        diversity_weight = max(0.0001, 0.001 - (epoch / 40))  # Much smaller weight
        diversity_loss = torch.mean(torch.sum(manager_output * worker_weights, dim=-1))
        
        # Wasserstein distance is the PRIMARY OBJECTIVE (not regularization)
        wass_loss = wass(manager_output, worker_weights)
        
        # Combine losses with proper signs:
        # - Minimize Wasserstein distance (manager should match worker weights) - PRIMARY OBJECTIVE
        # - Small entropy regularization (encourage exploration)
        # - Small diversity regularization (encourage different selections)
        manager_loss = 10 * wass_loss + 0.01 * kl_div  # Combine Wasserstein with KL divergence
        
        # Add L2 regularization with smaller weight
        l2_reg = 0.0
        for param in self.manager_model.parameters():
            l2_reg += torch.norm(param, p=2)
        manager_loss = manager_loss + 1e-8 * l2_reg
        
        # Check for NaN values
        if torch.isnan(manager_loss):
            print(f"Warning: NaN detected in manager loss.")
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
            
            # Use temperature-based sampling for better exploration
            temperature = max(0.1, 1.0 - (epoch / 20))  # Start with high temperature, decrease over time
            # if epoch < 5:  # Short exploration period
            #     # Apply temperature scaling
            #     scaled_probs = manager_probs / temperature
            #     # Sample from the distribution
            #     selected_workers = torch.multinomial(F.softmax(scaled_probs, dim=-1), 1).squeeze(-1)
            # else:
            # Use argmax for later epochs (more deterministic)
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

        # Log metrics (only on main process for distributed training)
        if self._is_main_process() and self.writer is not None:
            step = epoch * total_batches + batch_idx
            self.writer.add_scalar('Loss/worker', np.mean(worker_losses) if worker_losses else 0, step)
            self.writer.add_scalar('Loss/manager', manager_loss.item(), step)
            self.writer.add_scalar('Loss/entropy', entropy.item(), step)
            self.writer.add_scalar('Loss/entropy_weight', entropy_weight, step)
        
        return {
            'worker_loss': np.mean(worker_losses) if worker_losses else 0,
            'manager_loss': manager_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': entropy.item(),
            'entropy_weight': entropy_weight,
            'diversity_loss': diversity_loss.item(),
            'diversity_weight': diversity_weight,
            'wass_loss': wass_loss.item(),
            'worker_distribution': history,
            'worker_selections': selected_workers.cpu().numpy(),
            'active_workers': torch.unique(selected_workers).cpu().numpy()
        }

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
                selected_workers = torch.argmax(manager_output, dim=-1)  # Always use argmax for validation
                
                for i, (data, target) in enumerate(zip(worker_data, worker_target)):
                    worker_idx = selected_workers[i].item()
                    worker = self.worker_models[worker_idx]
                    
                    # WorkerWrapper handles missing temporal features automatically
                    # Pass both input data and target to the wrapper
                    output = worker(data.unsqueeze(0), target=target.unsqueeze(0))
                    loss = self.criterion(output, target.unsqueeze(0))
                    total_loss += loss.item()
                    total_samples += 1
        
        # Synchronize validation results across GPUs for distributed training
        if self.distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_samples_tensor = torch.tensor(total_samples, device=self.device)
            
            total_loss_tensor = self._sync_across_gpus(total_loss_tensor)
            total_samples_tensor = self._sync_across_gpus(total_samples_tensor)
            
            total_loss = total_loss_tensor.item()
            total_samples = int(total_samples_tensor.item())
        
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
        Load a checkpoint of the model with robust handling of architecture changes.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load manager model
        try:
            self.manager_model.load_state_dict(checkpoint['manager_state_dict'])
            # print("✓ Manager model loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load manager state dict: {e}")
            print("   Manager model will use random initialization")
        
        # Load worker models with robust error handling
        for i, (worker, state_dict) in enumerate(zip(self.worker_models, checkpoint['worker_state_dicts'])):
            try:
                # Try to load the state dict directly
                worker.load_state_dict(state_dict)
                # print(f"✓ Worker {i} loaded successfully")
            except RuntimeError as e:
                print(f"⚠️  Warning: Could not load worker {i} state dict directly: {e}")
                
                # Try to load with partial matching
                try:
                    self._load_partial_state_dict(worker, state_dict)
                    # print(f"✓ Worker {i} loaded with partial matching")
                except Exception as e2:
                    print(f"❌ Error: Could not load worker {i} even with partial matching: {e2}")
                    print("   Worker will use random initialization")
        
        # Load optimizers
        try:
            self.manager_optimizer.load_state_dict(checkpoint['manager_optimizer'])
            # print("✓ Manager optimizer loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load manager optimizer: {e}")
        
        for i, (opt, state_dict) in enumerate(zip(self.worker_optimizers, checkpoint['worker_optimizers'])):
            try:
                opt.load_state_dict(state_dict)
                # print(f"✓ Worker {i} optimizer loaded successfully")
            except Exception as e:
                print(f"  Warning: Could not load worker {i} optimizer: {e}")
        
        # Load training statistics
        try:
            self.train_counts = checkpoint['train_counts']
            self.total_count = checkpoint['total_count']
            # print("✓ Training statistics loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load training statistics: {e}")
            # Reset to defaults
            self.train_counts = torch.zeros(len(self.worker_models), device=self.device)
            self.total_count = 0
        
        return checkpoint['epoch'], checkpoint['best_val_loss']
    
    def _load_partial_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]):
        """
        Load state dict with partial matching to handle architecture changes.
        """
        model_state_dict = model.state_dict()
        
        # Create a new state dict with only matching keys
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            # Try to find a matching key in the current model
            if key in model_state_dict:
                # Direct match
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"   Shape mismatch for {key}: saved {value.shape} vs current {model_state_dict[key].shape}")
            else:
                # Try to find a similar key (handle architecture changes)
                matching_key = self._find_matching_key(key, model_state_dict.keys())
                if matching_key and model_state_dict[matching_key].shape == value.shape:
                    filtered_state_dict[matching_key] = value
                    print(f"   Mapped {key} -> {matching_key}")
        
        # Load the filtered state dict
        if filtered_state_dict:
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            print("   No compatible parameters found")
    
    def _find_matching_key(self, old_key: str, new_keys: List[str]) -> Optional[str]:
        """
        Find a matching key in the new model architecture.
        """
        # Handle common architecture changes
        key_mappings = {
            'enc_embedding': 'embedding',
            'dec_embedding': 'embedding',
            'decoder.projection': 'projection',
            'decoder.norm': 'norm'
        }
        
        for old_pattern, new_pattern in key_mappings.items():
            if old_pattern in old_key:
                new_key = old_key.replace(old_pattern, new_pattern)
                if new_key in new_keys:
                    return new_key
        
        # Try to find keys with similar structure
        old_parts = old_key.split('.')
        for new_key in new_keys:
            new_parts = new_key.split('.')
            if len(old_parts) == len(new_parts):
                # Check if most parts match
                matches = sum(1 for old, new in zip(old_parts, new_parts) if old == new)
                if matches >= len(old_parts) - 1:  # Allow one mismatch
                    return new_key
        
        return None

    def load_best_model(self, checkpoint_dir: str, force_fresh_start: bool = False) -> Tuple[int, float]:
        """
        Load the best model from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
            force_fresh_start: If True, skip loading and start fresh
            
        Returns:
            Tuple of (epoch, best_val_loss) from the checkpoint
        """
        if force_fresh_start:
            print("⚠️  Force fresh start requested - skipping checkpoint loading")
            return 0, float('inf')
            
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from {checkpoint_path}")
            try:
                return self.load_checkpoint(checkpoint_path)
            except Exception as e:
                print(f" Error loading checkpoint: {e}")
                print("  Starting with fresh model initialization")
                return 0, float('inf')
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")
            return 0, float('inf')

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: str = "checkpoints/catp",
        min_lr: float = 1e-5,
        plot_metrics: bool = True,
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None,
        pre_training_epochs: int = 0
    ) -> Dict[str, List[float]]:
        """
        General training loop for CATP model with multi-GPU support.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            min_lr: Minimum learning rate for cosine decay
            plot_metrics: Whether to plot metrics during training
            save_best_only: Whether to save only the best model
            early_stopping_patience: Number of epochs to wait before early stopping
            pre_training_epochs: Number of epochs to pre-train each worker model individually
            
        Returns:
            Dictionary containing training history
        """
        # Create checkpoint directory (only on main process)
        if self._is_main_process():
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Pre-training phase: train each worker model individually
        if pre_training_epochs > 0 and self._is_main_process():
            print(f"\n{'='*60}")
            print(f"PRE-TRAINING PHASE: Training each worker for {pre_training_epochs} epochs")
            print(f"{'='*60}")
            self._pre_train_workers(train_loader, val_loader, pre_training_epochs)
            print(f"\n{'='*60}")
            print(f"PRE-TRAINING COMPLETED. Starting CATP training...")
            print(f"{'='*60}\n")
        
        # Synchronize after pre-training
        if self.distributed:
            dist.barrier()
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        worker_selections = []
        no_improve_count = 0
        
        for epoch in range(epochs):
            # Set epoch for distributed sampler
            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            if self._is_main_process():
                print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Update learning rates with cosine decay
            current_manager_lr = self.manager_optimizer.param_groups[0]['lr']
            current_worker_lr = self.worker_optimizers[0].param_groups[0]['lr']
            
            # Update learning rates
            for param_group in self.manager_optimizer.param_groups:
                param_group['lr'] = current_manager_lr
            for optimizer in self.worker_optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_worker_lr
            
            if self._is_main_process():
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
                
                if self._is_main_process() and batch_idx % 32 == 0:
                    print(f"  Batch {batch_idx}: Worker Loss = {metrics['worker_loss']:.4f}, "
                          f"Manager Loss = {metrics['manager_loss']:.8f}")
                    
                    # Add detailed monitoring for first few batches
                    if batch_idx < 3:
                        print(f"    Debug - Loss Components:")
                        print(f"      KL Div: {metrics['kl_div']:.6f}, Entropy: {metrics['entropy']:.6f}")
                        print(f"      Diversity: {metrics['diversity_loss']:.6f}, Wasserstein: {metrics['wass_loss']:.6f}")
                        print(f"      Weights: entropy={metrics['entropy_weight']:.3f}, diversity={metrics['diversity_weight']:.3f}")
                        print(f"      Wasserstein (primary): {metrics['wass_loss']:.6f}")
            
            # Calculate epoch metrics
            train_loss = np.mean(epoch_train_losses)
            train_losses.append(train_loss)
            
            # Calculate worker selection rates
            epoch_selections = np.concatenate(epoch_worker_selections)
            selection_rates = np.bincount(epoch_selections, minlength=len(self.worker_models)) / len(epoch_selections)
            worker_selections.append(selection_rates)
            
            # Print worker selection rates for this epoch (only on main process)
            if self._is_main_process():
                print(f"Epoch {epoch + 1} Worker Selection Rates:")
                for i, rate in enumerate(selection_rates):
                    print(f"  Worker {i}: {rate:.3f}")
                print()
            
            # Validation
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            if self._is_main_process():
                print(f"Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint if validation loss improved (only on main process)
            if self._is_main_process() and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(checkpoint_path, epoch, best_val_loss)
                print(f"Saved checkpoint with validation loss: {best_val_loss:.4f}")
                no_improve_count = 0
            elif val_loss > best_val_loss and epoch > 20:
                no_improve_count += 1
            
            # Early stopping
            if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
                if self._is_main_process():
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Plot metrics if requested (only on main process)
            if self._is_main_process() and plot_metrics and (epoch + 1) % 5 == 0:
                self._plot_metrics(train_losses, val_losses, worker_selections)
        
        # Load the best model after training for evaluation (only on main process)
        # if self._is_main_process():
        #     print(f"\n{'='*60}")
        #     print("LOADING BEST MODEL FOR EVALUATION")
        #     print(f"{'='*60}")
            
        #     # Try to load the best model, but don't fail if there are issues
        #     try:
        #         best_epoch, best_val_loss = self.load_best_model(checkpoint_dir)
        #         print(f"Loaded best model from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        #     except Exception as e:
        #         print(f"❌ Error loading best model: {e}")
        #         print("⚠️  Using current model state for evaluation")
        #         best_epoch = len(train_losses) - 1 if train_losses else 0
        #         best_val_loss = val_losses[-1] if val_losses else float('inf')
        # else:
        #     # For non-main processes, use default values
        #     best_epoch = len(train_losses) - 1 if train_losses else 0
        #     best_val_loss = val_losses[-1] if val_losses else float('inf')
        
        # Synchronize before returning
        if self.distributed:
            dist.barrier()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'worker_selections': worker_selections,
            # 'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
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
    
    def _pre_train_workers(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pre_training_epochs: int
    ):
        """
        Pre-train each worker model individually for a specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            pre_training_epochs: Number of epochs to pre-train each worker
        """
        print(f"Starting pre-training for {len(self.worker_models)} workers...")
        
        for worker_idx in range(len(self.worker_models)):
            worker = self.worker_models[worker_idx]
            optimizer = self.worker_optimizers[worker_idx]
            
            print(f"\nPre-training Worker {worker_idx} ({type(worker.model).__name__})...")
            
            # Pre-training loop for this worker
            for epoch in range(pre_training_epochs):
                worker.train()
                epoch_losses = []
                
                # Training
                for batch_idx, batch_data in enumerate(train_loader):
                    worker_data, worker_target = batch_data
                    worker_data = worker_data.to(self.device)
                    worker_target = worker_target.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = worker(worker_data, target=worker_target)
                    loss = self.criterion(output, worker_target)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(worker.parameters(), self.clip_value)
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                    # if batch_idx % 50 == 0:
                    #     print(f"  Epoch {epoch + 1}/{pre_training_epochs}, Batch {batch_idx}: Loss = {loss.item():.4f}")
                
                # Validation for this worker
                worker.eval()
                val_loss = 0.0
                val_samples = 0
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        worker_data, worker_target = batch_data
                        worker_data = worker_data.to(self.device)
                        worker_target = worker_target.to(self.device)
                        
                        output = worker(worker_data, target=worker_target)
                        loss = self.criterion(output, worker_target)
                        val_loss += loss.item() * worker_data.size(0)
                        val_samples += worker_data.size(0)
                
                avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
                avg_train_loss = np.mean(epoch_losses)
                
                print(f"  Epoch {epoch + 1}/{pre_training_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            print(f"Worker {worker_idx} pre-training completed!")
        
        print(f"\nAll workers pre-training completed!")
        
        # Reset training counts after pre-training
        self.train_counts = torch.zeros(len(self.worker_models), device=self.device)
        self.total_count = 0 

    def _print_gpu_info(self):
        """Print information about available GPUs."""
        if torch.cuda.is_available():
            print(f" GPU Information:")
            print(f"   Total GPUs: {torch.cuda.device_count()}")
            print(f"   Current device: {self.device}")
            print(f"   Multi-GPU enabled: {self.use_multi_gpu}")
            print(f"   Distributed training: {self.distributed}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print(" Using CPU for training")
    
    def _is_main_process(self) -> bool:
        """Check if this is the main process (for distributed training)."""
        return not self.distributed or self.rank == 0
    
    def _sync_across_gpus(self, tensor: torch.Tensor) -> torch.Tensor:
        """Synchronize tensor across all GPUs in distributed training."""
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
        return tensor
    
    def _create_distributed_sampler(self, dataset, shuffle: bool = True):
        """Create a distributed sampler for the dataset."""
        if self.distributed:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
        return None 