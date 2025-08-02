# SOTA Frontier: A Plan for a State-Space Model-Based Hierarchical Architecture

This document outlines a literature review and implementation plan for exploring the state-of-the-art (SOTA) frontier in sequence modeling, with the goal of designing a next-generation hierarchical reasoning model.

## 1. Literature Review: The Rise of State-Space Models

The current SOTA in sequence modeling, particularly for long sequences, is dominated by **Structured State-Space Models (SSMs)**. This family of architectures, most notably **Mamba**, has emerged as a powerful and efficient alternative to the Transformer.

### Key Papers & Concepts:

1.  **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"** (Gu & Dao, 2023): This is the foundational paper for Mamba.
    *   **Core Idea**: It introduces a *selection mechanism* that allows the model to selectively propagate or forget information based on the input context. This gives an SSM the ability to perform content-aware reasoning, a key strength of Transformers.

2.  **"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"** (Dao & Gu, 2024): This paper introduces Mamba-2 and establishes a theoretical link between SSMs and attention.
    *   **Core Idea**: It shows that the attention mechanism is a specific form of a more general class of models that includes SSMs. This insight allows for transferring optimizations from the Transformer world to SSMs.

3.  **Surveys and Application Papers**: Numerous recent papers (e.g., "A Survey of Mamba", "A Survey on Vision Mamba") review the rapid adoption of Mamba across various domains, from language to computer vision and time-series forecasting. Papers like "CryptoMamba" and "MambaLLM" demonstrate its effectiveness in financial prediction tasks.

### Why is Mamba Better than HRM?

A direct academic comparison is not available, but we can infer the architectural superiority of Mamba over HRM based on their designs:

| Feature | Hierarchical Reasoning Model (HRM) | Mamba (SSM) | Advantage |
| :--- | :--- | :--- | :--- |
| **Memory Complexity** | `O(1)` per cycle, but by **truncating gradients**. | `O(1)` state during inference, `O(L*D)` for training (linear). | **Mamba**. It doesn't need to "forget" its history to be efficient. |
| **Gradient Flow** | Myopic. Gradients only flow through the last step of a cycle. | Global. Gradients flow through the entire sequence. | **Mamba**. It allows for true long-range dependency learning. |
| **Parallelization** | Limited. The recurrent cycles are inherently sequential. | Highly parallelizable during training (convolutional mode). | **Mamba**. Leads to much faster training. |
| **Underlying Theory** | Heuristic, brain-inspired design. | Principled, based on control theory and linear systems. | **Mamba**. More robust and theoretically grounded. |
| **Inference** | Fast and efficient (`O(1)` state per cycle). | Extremely fast and efficient (`O(1)` state per step). | **Draw**. Both are very efficient at inference, but Mamba is more general. |

In short, Mamba solves the long-sequence problem in a more fundamental and effective way than HRM. While HRM uses a clever trick (gradient truncation) to achieve efficiency, Mamba provides a new, powerful modeling paradigm that is both efficient and expressive without sacrificing the ability to learn from the full sequence history.

## 2. Implementation Review: Available Codebases

The clear winner for a foundational codebase is the official implementation from the authors.

*   **Official Repository**: **`state-spaces/mamba`** on GitHub.
    *   **Link**: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
    *   **Why it's the best choice**:
        *   It is the reference implementation from the creators of Mamba.
        *   It includes the core `mamba-ssm` package, which can be installed via pip.
        *   It provides pretrained language models of various sizes (from 130M to 2.8B parameters) hosted on Hugging Face, which is invaluable for transfer learning.
        *   It contains benchmarking scripts for comparing performance against other architectures like Transformers (e.g., Pythia).
        *   The code is actively maintained and has a large community.

*   **Hugging Face Hub**: **`state-spaces`** organization.
    *   **Link**: [https://huggingface.co/state-spaces](https://huggingface.co/state-spaces)
    *   **Contents**: This hub contains the pretrained model weights, which can be easily loaded using the `transformers` library or the official Mamba codebase.

## 3. Implementation Plan: Building a Hierarchical Mamba

Our goal is to create a new hierarchical model that benefits from the insights of both HRM and the power of Mamba. Instead of HRM's ad-hoc recurrent structure, we will use Mamba blocks as the fundamental building blocks for our reasoning modules.

### Phase 1: Foundational Mamba Block (Weeks 1-2)

1.  **Setup Environment**:
    *   Clone the `state-spaces/mamba` repository.
    *   Install the `mamba-ssm` package and its dependencies (PyTorch, causal-conv1d, etc.).
2.  **Familiarize with the Core Module**:
    *   Focus on `mamba_ssm.modules.mamba_simple.Mamba`. This is the main Mamba block.
    *   Write a simple script to pass a random tensor through a single Mamba block to understand its inputs and outputs.
3.  **Benchmark Baseline**:
    *   Run the provided `benchmarks/benchmark_generation_mamba_simple.py` script to load a pretrained Mamba model (e.g., `mamba-130m`) and generate text. This validates that the environment is working correctly and provides a performance baseline.

### Phase 2: Design and Implement the Hierarchical Mamba (H-Mamba) (Weeks 3-4)

1.  **Architectural Design**:
    *   Define a new `H-Mamba` model class. This class will not be recurrent in the same way as HRM. Instead, it will be a deep, feed-forward model that *simulates* hierarchy through its structure.
    *   **Low-Level Module (L-Mamba)**: A stack of several Mamba blocks that process the initial input sequence.
    *   **High-Level Module (H-Mamba)**: Another stack of Mamba blocks that takes the output of the L-Mamba module as its input.
    *   **Input/Output**: The model will take a sequence of tokens as input and produce a sequence of logits as output, just like a standard language model.

2.  **Implementation**:
    *   Create a new file, e.g., `models/h_mamba.py`.
    *   Implement the `H_Mamba` class using `torch.nn.Module`.
    *   Instantiate two sets of Mamba blocks (e.g., `self.l_level = nn.Sequential(*[Mamba(...) for _ in range(L_layers)])` and `self.h_level = nn.Sequential(...)`).
    *   The `forward` method will simply be `h_out = self.h_level(self.l_level(x))`.

### Phase 3: Training and Evaluation (Weeks 5-8)

1.  **Prepare a Training Task**:
    *   We can start with a simple task like character-level language modeling on a small dataset (e.g., TinyShakespeare) to validate that the model can learn.
    *   This will involve setting up a standard PyTorch training loop (data loader, optimizer, loss function).
2.  **Benchmark on Standard Tasks**:
    *   Adapt the evaluation scripts from the Mamba repository (e.g., `evals/lm_harness_eval.py`) to work with our new `H-Mamba` architecture.
    *   Fine-tune a pretrained Mamba model (by loading its weights into our L- and H-modules) or train our `H-Mamba` from scratch on a larger dataset like the Pile.
3.  **Analyze and Iterate**:
    *   Compare the performance (perplexity, zero-shot accuracy on benchmarks) of `H-Mamba` against a standard flat Mamba model with the same number of total layers.
    *   **Key Question to Answer**: Does explicitly structuring the model into a "low-level" and "high-level" block provide any performance benefit over a simple, deep stack of Mamba blocks? We may find that the hierarchical structure is less important than the raw power of the Mamba block itself.
    *   Experiment with different configurations (e.g., number of layers in L vs. H, adding pooling or down-sampling between the modules).

This plan provides a clear path from understanding the SOTA to implementing and evaluating a novel architecture that combines the hierarchical concepts of HRM with the superior efficiency and performance of Mamba.

