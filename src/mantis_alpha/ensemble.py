import numpy as np
import torch

class TemporalEnsemble:
    """
    Temporal Action Ensembling over overlapping action chunks (ACT-style).
    Maintains a rolling buffer of predicted future action trajectories and computes
    an exponential decay weighted average across predictions for smooth, jitter-free execution.
    """
    def __init__(self, chunk_size: int = 50, weight_decay: float = 0.95, device: str = "cpu"):
        self.chunk_size = chunk_size
        self.weight_decay = weight_decay
        self.device = device
        self.reset()
        
    def reset(self):
        """Clear action buffer for a new episode."""
        self.actions_buffer = []  # List of (start_step, action_chunk)
        self.step_idx = 0
        
    def update(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Input:
            action_chunk: Tensor of shape (chunk_size, action_dim) or (1, chunk_size, action_dim)
        Output:
            single action step for current timestep: Tensor of shape (action_dim,)
        """
        if action_chunk.ndim == 3:
            action_chunk = action_chunk.squeeze(0)
            
        action_chunk = action_chunk.detach().to(self.device)
        self.actions_buffer.append((self.step_idx, action_chunk))
        
        # Aggregate overlapping predictions for current step
        curr_actions = []
        weights = []
        
        for start_step, chunk in self.actions_buffer:
            offset = self.step_idx - start_step
            if 0 <= offset < self.chunk_size:
                pred_action = chunk[offset]
                weight = np.exp(-self.weight_decay * offset)
                curr_actions.append(pred_action)
                weights.append(weight)
                
        # Remove chunks that have expired
        self.actions_buffer = [
            (st, ch) for st, ch in self.actions_buffer 
            if (self.step_idx - st) < self.chunk_size
        ]
        
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        weights = weights / weights.sum()
        
        stacked_actions = torch.stack(curr_actions, dim=0)
        ensembled_action = (stacked_actions * weights.unsqueeze(-1)).sum(dim=0)
        
        self.step_idx += 1
        return ensembled_action
