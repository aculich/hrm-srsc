# Connecting Hierarchical Reasoning Model with Square-Root Space Complexity

## Executive Summary

This analysis explores the potential connections between the **Hierarchical Reasoning Model (HRM)** from [Wang et al., 2025](https://arxiv.org/abs/2506.21734) and the **square-root space simulation** ideas from the STOC 2025 paper "Simulating Time With Square-Root Space". Both papers address fundamental computational efficiency challenges, and their intersection could lead to novel improvements in neural reasoning architectures.

## 🧠 HRM Architecture Overview

### Core Design Principles

The HRM introduces a brain-inspired architecture with:

1. **Two Recurrent Modules**:
   - **High-level (H) module**: Slow, abstract planning (like prefrontal cortex)
   - **Low-level (L) module**: Rapid, detailed computations (like sensory cortex)

2. **Hierarchical Convergence**:
   - L-module converges quickly within each cycle
   - H-module evolves slowly across multiple cycles
   - Avoids premature convergence of standard RNNs

3. **Memory Efficiency**:
   - Constant memory footprint (O(1) instead of O(T))
   - No need for BPTT (Backpropagation Through Time)
   - 27M parameters achieving state-of-the-art results

### Key Innovation: Implicit Gradient Truncation

```python
# Pseudo-code for HRM's gradient computation
def hrm_forward(x, cycles):
    h_state = init_high_level()
    for cycle in range(cycles):
        # L-module converges quickly
        l_state = init_low_level()
        for t in range(convergence_steps):
            l_state = f_L(x, l_state, h_state)
        
        # H-module updates slowly
        h_state = f_H(h_state, l_state)
        
        # Implicit gradient truncation at cycle boundaries
        if training:
            compute_gradients(local_only=True)
```

## 📊 Square-Root Space Complexity Insights

### Main Theorem (Simplified)

The square-root space paper proves:

**Theorem 1.1**: For deterministic algorithms:
- DTISP(T, S) ⊆ DTISP(T·polylog(T), O(√(T·S) + S))
- Time T with space S can be simulated with time T·polylog(T) and space O(√(T·S))

### Key Techniques

1. **Checkpointing Strategy**:
   - Store checkpoints at √T intervals
   - Recompute between checkpoints as needed
   - Trade time for space efficiency

2. **Space-Time Tradeoffs**:
   - Balance between computation and memory
   - Optimal for certain complexity classes
   - Robust across different computational models

## 🔗 Potential Connections to HRM

### 1. Hierarchical Checkpointing in Neural Networks

The HRM's two-level hierarchy naturally aligns with checkpointing:

```python
class HRM_with_Checkpointing:
    def __init__(self):
        self.checkpoint_interval = int(sqrt(self.max_cycles))
        self.checkpoints = {}
    
    def forward(self, x, cycles):
        h_state = self.init_high_level()
        
        for cycle in range(cycles):
            # Store checkpoint at sqrt intervals
            if cycle % self.checkpoint_interval == 0:
                self.checkpoints[cycle] = h_state.detach()
            
            # Regular HRM computation
            l_state = self.run_low_level(x, h_state)
            h_state = self.update_high_level(h_state, l_state)
        
        return h_state
    
    def backward_with_recomputation(self, grad_output):
        # Recompute forward from nearest checkpoint
        # Saves memory at cost of computation
        pass
```

### 2. Memory-Efficient Training

Apply square-root space principles to HRM training:

**Current HRM**: O(1) memory per cycle
**With sqrt-space**: O(√T) memory for T cycles with full gradient information

```python
def sqrt_space_training(model, data, T_cycles):
    # Store gradients at sqrt(T) checkpoints
    checkpoint_grads = {}
    checkpoint_interval = int(sqrt(T_cycles))
    
    for epoch in range(num_epochs):
        for cycle in range(T_cycles):
            # Forward pass
            output = model.forward_cycle(data, cycle)
            
            # Store gradients sparsely
            if cycle % checkpoint_interval == 0:
                loss = compute_loss(output)
                grad = autograd.grad(loss, model.parameters())
                checkpoint_grads[cycle] = grad
            
            # Periodic gradient accumulation
            if cycle % checkpoint_interval == checkpoint_interval - 1:
                apply_accumulated_gradients(model, checkpoint_grads)
```

### 3. Adaptive Computation Time with Space Bounds

Combine HRM's ACT mechanism with space complexity bounds:

```python
class SpaceBoundedACT:
    def __init__(self, max_space_sqrt):
        self.max_checkpoints = max_space_sqrt
        self.active_checkpoints = []
    
    def should_halt(self, halt_prob, cycle):
        # Halt if probability high OR space limit reached
        space_constraint = len(self.active_checkpoints) >= self.max_checkpoints
        return halt_prob > 0.9 or space_constraint
```

## 💡 Implementation Ideas for HRM Codebase

### 1. Checkpoint-Based Memory Management

**File**: `models/hrm/hrm_act_v1.py`

Add checkpointing to the HRM forward pass:

```python
class HierarchicalReasoningModel_ACTV1_Checkpointed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_checkpointing = config.get('use_sqrt_checkpointing', False)
        self.checkpoint_interval = int(math.sqrt(config.L_cycles))
        
    def forward(self, x, cycles):
        if self.use_checkpointing:
            return self._forward_with_checkpoints(x, cycles)
        else:
            return self._forward_standard(x, cycles)
    
    def _forward_with_checkpoints(self, x, cycles):
        checkpoints = {}
        h_state = self.h_init
        
        for cycle in range(cycles):
            # Checkpoint high-level state
            if cycle % self.checkpoint_interval == 0:
                checkpoints[cycle] = h_state.detach().clone()
            
            # Standard HRM computation
            l_state = self.l_module(x, h_state)
            h_state = self.h_module(h_state, l_state)
            
        return h_state, checkpoints
```

### 2. Gradient Accumulation Strategy

**File**: `pretrain.py`

Modify training loop for square-root space gradient accumulation:

```python
def train_with_sqrt_space(model, data_loader, config):
    """Training with square-root space complexity for gradients"""
    
    sqrt_accumulation_steps = int(math.sqrt(config.L_cycles))
    gradient_checkpoints = {}
    
    for batch in data_loader:
        # Forward pass with checkpointing
        output, checkpoints = model(batch, use_checkpointing=True)
        
        # Compute loss at checkpoints only
        for cp_cycle, cp_state in checkpoints.items():
            loss = compute_loss_at_checkpoint(cp_state, batch.labels)
            
            # Store gradients instead of applying immediately
            grads = torch.autograd.grad(loss, model.parameters(), 
                                       retain_graph=True,
                                       create_graph=False)
            gradient_checkpoints[cp_cycle] = grads
        
        # Apply accumulated gradients
        if len(gradient_checkpoints) >= sqrt_accumulation_steps:
            apply_sqrt_accumulated_gradients(model, gradient_checkpoints)
            gradient_checkpoints.clear()
```

### 3. Memory-Bounded Reasoning

**File**: `models/hrm/hrm_act_v1.py`

Add space complexity constraints to ACT:

```python
class MemoryBoundedACT(nn.Module):
    def __init__(self, hidden_size, max_memory_sqrt):
        super().__init__()
        self.halt_predictor = nn.Linear(hidden_size, 1)
        self.max_memory_sqrt = max_memory_sqrt
        self.stored_states = []
        
    def forward(self, states, cycle):
        # Compute halt probability
        halt_logits = self.halt_predictor(states)
        halt_prob = torch.sigmoid(halt_logits)
        
        # Check memory constraint
        memory_used = len(self.stored_states)
        if memory_used >= self.max_memory_sqrt:
            # Force halt due to memory limit
            return torch.ones_like(halt_prob)
        
        # Store state at sqrt intervals
        if cycle % int(math.sqrt(self.max_memory_sqrt)) == 0:
            self.stored_states.append(states.detach())
        
        return halt_prob
```

### 4. Hierarchical Memory Pooling

Implement square-root space memory pooling for long sequences:

```python
class SqrtMemoryPool:
    """Memory pool with O(sqrt(T)) space complexity"""
    
    def __init__(self, max_sequence_length):
        self.pool_size = int(math.sqrt(max_sequence_length))
        self.memory_pool = [None] * self.pool_size
        self.access_pattern = []
        
    def store(self, timestep, state):
        # Hash timestep to pool location
        pool_idx = timestep % self.pool_size
        self.memory_pool[pool_idx] = (timestep, state)
        self.access_pattern.append(timestep)
        
    def retrieve(self, timestep):
        # Check if in pool
        pool_idx = timestep % self.pool_size
        if self.memory_pool[pool_idx] and self.memory_pool[pool_idx][0] == timestep:
            return self.memory_pool[pool_idx][1]
        
        # Otherwise, recompute from nearest checkpoint
        nearest_checkpoint = self.find_nearest_checkpoint(timestep)
        return self.recompute_from_checkpoint(nearest_checkpoint, timestep)
```

## 🔬 Experimental Validation

### Proposed Experiments

1. **Memory Usage Analysis**:
   - Compare standard HRM vs. sqrt-checkpointed HRM
   - Measure peak memory during training
   - Track memory-accuracy tradeoffs

2. **Convergence Speed**:
   - Test if checkpointing affects hierarchical convergence
   - Measure cycles to convergence with different checkpoint intervals

3. **Scalability Tests**:
   - Train on longer sequences (10x current length)
   - Compare performance with/without sqrt-space techniques

### Metrics to Track

```python
@dataclass
class SqrtSpaceMetrics:
    peak_memory_mb: float
    checkpoint_count: int
    recomputation_time_ms: float
    accuracy_delta: float
    convergence_cycles: int
```

## 🚀 Implementation Roadmap

### Phase 1: Basic Checkpointing (Week 1-2)
- [ ] Add checkpoint storage to HRM forward pass
- [ ] Implement gradient recomputation from checkpoints
- [ ] Validate correctness on small examples

### Phase 2: Memory-Bounded Training (Week 3-4)
- [ ] Integrate sqrt-space gradient accumulation
- [ ] Add memory monitoring and constraints
- [ ] Benchmark memory usage vs. standard training

### Phase 3: Advanced Features (Week 5-6)
- [ ] Implement adaptive checkpoint intervals
- [ ] Add hierarchical memory pooling
- [ ] Optimize recomputation strategies

### Phase 4: Evaluation (Week 7-8)
- [ ] Run full experiments on ARC, Sudoku, Maze tasks
- [ ] Compare with baseline HRM
- [ ] Prepare performance analysis

## 📚 Theoretical Insights

### Connection to Brain's Memory Systems

1. **Working Memory Capacity**: Human working memory holds ~7±2 items, similar to O(√n) scaling
2. **Episodic Memory**: Brain uses sparse encoding, retrieving full episodes from partial cues
3. **Hierarchical Processing**: Cortical hierarchy naturally implements space-time tradeoffs

### Computational Advantages

1. **Longer Sequences**: Train on sequences 10-100x longer with same memory
2. **Larger Models**: Scale to larger hidden dimensions with bounded memory
3. **Online Learning**: Enable continual learning with bounded memory growth

## 🎯 Conclusion

The square-root space complexity results offer promising directions for enhancing HRM:

1. **Theoretical**: Provides formal guarantees on memory-computation tradeoffs
2. **Practical**: Enables training on longer sequences and larger models
3. **Biological**: Aligns with known constraints of biological neural systems

By incorporating these ideas, HRM could achieve:
- **10x longer sequence processing** with same memory budget
- **Provable memory bounds** for deployment scenarios
- **Better alignment** with biological memory constraints

The key insight is that HRM's hierarchical structure naturally supports checkpointing strategies, making it an ideal candidate for square-root space optimizations.

---

**Next Steps**: 
1. Implement basic checkpointing in HRM codebase
2. Run memory profiling experiments
3. Submit findings to NeurIPS 2025 workshop on Memory-Efficient Deep Learning