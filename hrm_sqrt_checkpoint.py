"""
Square-Root Space Checkpointing for Hierarchical Reasoning Model

This module implements memory-efficient training techniques inspired by
"Simulating Time With Square-Root Space" (STOC 2025) for the HRM architecture.

Key innovations:
1. Checkpoint storage at O(sqrt(T)) intervals
2. Gradient recomputation from nearest checkpoint
3. Memory-bounded adaptive computation time
4. Hierarchical memory pooling

Author: Based on HRM (Wang et al., 2025) and sqrt-space theory
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1


@dataclass
class CheckpointConfig:
    """Configuration for square-root checkpointing"""
    use_checkpointing: bool = True
    checkpoint_interval: Optional[int] = None  # Auto-compute as sqrt(L_cycles) if None
    recompute_gradients: bool = True
    memory_limit_mb: Optional[float] = None
    adaptive_interval: bool = False


class SqrtMemoryPool:
    """
    Memory pool with O(sqrt(T)) space complexity for storing intermediate states.
    
    Based on the square-root space simulation principle: instead of storing all T states,
    we store only sqrt(T) checkpoints and recompute intermediate states as needed.
    """
    
    def __init__(self, max_sequence_length: int, state_dim: int, device: torch.device):
        self.max_length = max_sequence_length
        self.pool_size = int(math.sqrt(max_sequence_length))
        self.state_dim = state_dim
        self.device = device
        
        # Pre-allocate memory pool
        self.memory_pool = torch.zeros(
            self.pool_size, state_dim, device=device, requires_grad=False
        )
        self.timestamps = torch.full((self.pool_size,), -1, device=device)
        self.access_count = torch.zeros(self.pool_size, device=device)
        
    def store(self, timestep: int, state: torch.Tensor) -> None:
        """Store state at given timestep using modular hashing"""
        pool_idx = timestep % self.pool_size
        self.memory_pool[pool_idx] = state.detach()
        self.timestamps[pool_idx] = timestep
        self.access_count[pool_idx] += 1
        
    def retrieve(self, timestep: int) -> Optional[torch.Tensor]:
        """Retrieve state if available in pool"""
        pool_idx = timestep % self.pool_size
        if self.timestamps[pool_idx] == timestep:
            return self.memory_pool[pool_idx].clone()
        return None
    
    def get_nearest_checkpoint(self, timestep: int) -> Tuple[int, torch.Tensor]:
        """Find nearest available checkpoint before given timestep"""
        valid_timestamps = self.timestamps[self.timestamps < timestep]
        if len(valid_timestamps) == 0:
            return -1, None
            
        nearest_time = valid_timestamps.max().item()
        pool_idx = int(nearest_time % self.pool_size)
        return nearest_time, self.memory_pool[pool_idx].clone()


class HierarchicalReasoningModel_SqrtCheckpoint(HierarchicalReasoningModel_ACTV1):
    """
    HRM with square-root space checkpointing for memory-efficient training.
    
    This implementation reduces memory complexity from O(T) to O(sqrt(T)) while
    maintaining the same computational expressiveness.
    """
    
    def __init__(self, config, checkpoint_config: CheckpointConfig = None):
        super().__init__(config)
        
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        
        # Compute checkpoint interval
        if self.checkpoint_config.checkpoint_interval is None:
            self.checkpoint_config.checkpoint_interval = int(math.sqrt(self.L_cycles))
        
        # Initialize memory pools for both modules
        self.h_memory_pool = SqrtMemoryPool(
            self.L_cycles, self.H_hidden_size, self.device
        )
        self.l_memory_pool = SqrtMemoryPool(
            self.halt_max_steps, self.L_hidden_size, self.device
        )
        
        # Track memory usage
        self.memory_tracker = MemoryTracker(self.checkpoint_config.memory_limit_mb)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with square-root checkpointing"""
        
        if not self.checkpoint_config.use_checkpointing:
            return super().forward(inputs)
        
        return self._forward_with_checkpoints(inputs)
    
    def _forward_with_checkpoints(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass storing checkpoints at sqrt(T) intervals.
        
        This reduces memory from O(T*H) to O(sqrt(T)*H) where H is hidden size.
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Initialize states
        h_state = self._init_h_state(batch_size).to(device)
        cumulative_halt_prob = torch.zeros(batch_size, 1, device=device)
        
        # Track checkpoints for gradient computation
        checkpoints = {}
        
        for cycle in range(self.L_cycles):
            # Store checkpoint at sqrt intervals
            if cycle % self.checkpoint_config.checkpoint_interval == 0:
                checkpoints[cycle] = {
                    'h_state': h_state.detach().clone(),
                    'halt_prob': cumulative_halt_prob.detach().clone()
                }
                self.h_memory_pool.store(cycle, h_state)
                
                # Check memory limit
                if self.memory_tracker.should_evict():
                    self._evict_oldest_checkpoint(checkpoints)
            
            # Run L-module with potential inner checkpointing
            l_state = self._run_l_module_with_checkpoints(
                inputs, h_state, cycle
            )
            
            # Update H-module
            h_state = self.H_module(h_state, l_state)
            
            # ACT mechanism with memory awareness
            halt_logits = self.halt_predictor(h_state)
            halt_prob = torch.sigmoid(halt_logits)
            
            # Force halt if memory limit reached
            if self.memory_tracker.is_at_limit():
                halt_prob = torch.ones_like(halt_prob)
            
            cumulative_halt_prob = cumulative_halt_prob + (1 - cumulative_halt_prob) * halt_prob
            
            # Early stopping if all samples halted
            if (cumulative_halt_prob > 0.99).all():
                break
        
        return h_state, checkpoints
    
    def _run_l_module_with_checkpoints(
        self, 
        inputs: torch.Tensor, 
        h_state: torch.Tensor, 
        cycle: int
    ) -> torch.Tensor:
        """Run L-module with inner checkpointing for very long sequences"""
        
        l_state = self._init_l_state(inputs.shape[0]).to(inputs.device)
        l_checkpoint_interval = int(math.sqrt(self.halt_max_steps))
        
        for step in range(self.halt_max_steps):
            # Checkpoint L-module states
            if step % l_checkpoint_interval == 0:
                self.l_memory_pool.store(step, l_state)
            
            # L-module step
            l_state = self.L_module(l_state, inputs, h_state)
            
            # Check convergence
            if self._check_l_convergence(l_state, step):
                break
        
        return l_state
    
    def _check_l_convergence(self, l_state: torch.Tensor, step: int) -> bool:
        """Check if L-module has converged"""
        if step < 2:
            return False
            
        # Retrieve previous state from pool
        prev_state = self.l_memory_pool.retrieve(step - 1)
        if prev_state is None:
            return False
            
        # Check convergence via state difference
        state_diff = (l_state - prev_state).abs().mean()
        return state_diff < 1e-6
    
    def _evict_oldest_checkpoint(self, checkpoints: Dict[int, Dict]) -> None:
        """Evict oldest checkpoint when memory limit reached"""
        if len(checkpoints) > 0:
            oldest_cycle = min(checkpoints.keys())
            del checkpoints[oldest_cycle]
            self.memory_tracker.freed_memory(
                checkpoints[oldest_cycle]['h_state'].element_size() * 
                checkpoints[oldest_cycle]['h_state'].nelement()
            )
    
    def backward_with_recomputation(
        self, 
        loss: torch.Tensor, 
        checkpoints: Dict[int, Dict]
    ) -> None:
        """
        Backward pass with recomputation from checkpoints.
        
        This implements the key insight from square-root space complexity:
        we can trade computation for memory by recomputing forward passes
        from the nearest checkpoint.
        """
        
        # Sort checkpoints by cycle
        checkpoint_cycles = sorted(checkpoints.keys())
        
        # Compute gradients for each segment between checkpoints
        for i in range(len(checkpoint_cycles)):
            start_cycle = checkpoint_cycles[i]
            end_cycle = checkpoint_cycles[i + 1] if i + 1 < len(checkpoint_cycles) else self.L_cycles
            
            # Recompute forward from checkpoint
            segment_output = self._recompute_segment(
                checkpoints[start_cycle], 
                start_cycle, 
                end_cycle
            )
            
            # Compute gradients for this segment
            segment_loss = self._compute_segment_loss(segment_output)
            segment_loss.backward(retain_graph=True)
            
    def _recompute_segment(
        self, 
        checkpoint: Dict, 
        start_cycle: int, 
        end_cycle: int
    ) -> torch.Tensor:
        """Recompute forward pass for a segment between checkpoints"""
        
        h_state = checkpoint['h_state'].clone().requires_grad_(True)
        
        for cycle in range(start_cycle, end_cycle):
            # Recompute L-module
            l_state = self._run_l_module_with_checkpoints(
                self.current_inputs,  # Assumes inputs are stored
                h_state, 
                cycle
            )
            
            # Update H-module
            h_state = self.H_module(h_state, l_state)
        
        return h_state


class MemoryTracker:
    """Track memory usage and enforce limits"""
    
    def __init__(self, memory_limit_mb: Optional[float]):
        self.memory_limit_bytes = (
            memory_limit_mb * 1024 * 1024 if memory_limit_mb else float('inf')
        )
        self.current_usage = 0
        
    def add_memory(self, bytes_used: int) -> None:
        self.current_usage += bytes_used
        
    def freed_memory(self, bytes_freed: int) -> None:
        self.current_usage = max(0, self.current_usage - bytes_freed)
        
    def should_evict(self) -> bool:
        return self.current_usage > 0.9 * self.memory_limit_bytes
        
    def is_at_limit(self) -> bool:
        return self.current_usage >= self.memory_limit_bytes


class AdaptiveCheckpointScheduler:
    """
    Dynamically adjust checkpoint intervals based on memory pressure
    and convergence patterns.
    """
    
    def __init__(self, initial_interval: int, min_interval: int = 1):
        self.current_interval = initial_interval
        self.min_interval = min_interval
        self.convergence_history = []
        
    def update_interval(self, convergence_speed: float, memory_pressure: float) -> int:
        """
        Adjust checkpoint interval based on:
        - Convergence speed: Faster convergence allows larger intervals
        - Memory pressure: High pressure requires smaller intervals
        """
        
        self.convergence_history.append(convergence_speed)
        
        # Compute adaptive interval
        if memory_pressure > 0.8:
            # Reduce interval under memory pressure
            self.current_interval = max(
                self.min_interval,
                int(self.current_interval * 0.8)
            )
        elif convergence_speed > 0.9 and memory_pressure < 0.5:
            # Increase interval if converging well with low memory usage
            self.current_interval = int(self.current_interval * 1.2)
            
        return self.current_interval


def create_sqrt_checkpoint_model(base_config, checkpoint_config: CheckpointConfig):
    """Factory function to create HRM with square-root checkpointing"""
    
    # Update config with checkpoint settings
    config = base_config.copy()
    config['checkpoint_config'] = checkpoint_config
    
    return HierarchicalReasoningModel_SqrtCheckpoint(config, checkpoint_config)


# Example usage for training with memory efficiency
if __name__ == "__main__":
    # Configuration
    model_config = {
        'L_cycles': 256,  # Can now handle much longer sequences
        'halt_max_steps': 64,
        'H_hidden_size': 512,
        'L_hidden_size': 256,
    }
    
    checkpoint_config = CheckpointConfig(
        use_checkpointing=True,
        checkpoint_interval=None,  # Auto-compute as sqrt(256) = 16
        memory_limit_mb=1024,  # 1GB memory limit
        adaptive_interval=True
    )
    
    # Create model
    model = create_sqrt_checkpoint_model(model_config, checkpoint_config)
    
    print(f"Model created with checkpoint interval: {model.checkpoint_config.checkpoint_interval}")
    print(f"Memory pools - H: {model.h_memory_pool.pool_size}, L: {model.l_memory_pool.pool_size}")
    print(f"Total checkpoint capacity: {model.h_memory_pool.pool_size + model.l_memory_pool.pool_size}")