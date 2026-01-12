# Not All Sequence Model Created Equal: Revealing the Inductive Biases of Sequence Models via Elementary Cellular Automata

## **Abstract**

The recent proliferation of "reasoning" models has largely been evaluated on natural language benchmarks (e.g., GSM8K, MATH). While effective for general capabilities, these benchmarks conflate linguistic competence with algorithmic reasoning. To isolate the latter, we introduce **ECA-Zero**, a synthetic dataset based on Elementary Cellular Automata (ECA) designed to test Deduction, Induction, and Abduction under strict Chain-of-Thought supervision. We evaluate a diverse suite of modern sequence architectures—including Standard Transformers, Liquid Foundation Models (LFM2), Multi-Head Latent Attention (MLA), Kimi Delta Attention (KDA), PaTH-FoX, and MesaNet. Preliminary results reveal striking disparities in how different architectures handle algorithmic complexity. Notably, hybrid architectures utilizing local convolutions (LFM2) appear to learn forward simulation (Deduction) significantly faster than pure Transformers, while dynamic state models (PaTH-FoX) show unexpected dominance in rule discovery (Induction). This paper presents our methodology and initial findings, with comprehensive evaluations of sparse and memory-routed architectures ongoing.

---

### 1. Introduction

The ability of Large Language Models (LLMs) to perform "reasoning" is often attributed to the scaling properties of the Transformer architecture. However, as alternative architectures—such as Linear Attention, Recurrent Neural Networks (RNNs), and Hybrid models—emerge, it remains unclear whether the "Transformer" is the optimal inductive bias for algorithmic tasks, or simply the most successfully scaled one.

We propose that natural language is a noisy medium for evaluating pure algorithmic reasoning. To rigorously test the inductive biases of these architectures, we turn to **Elementary Cellular Automata (ECA)**. ECA rules (0-255) provide a controlled, deterministic environment that is computationally universal (Turing complete via Rule 110) yet locally defined.

By training models on ECA tasks ranging from trivial (Class 1) to chaotic (Class 3) and complex (Class 4), we aim to answer:
1.  Does the $O(N^2)$ global attention mechanism provide a necessary advantage for local state transitions?
2.  Do Hybrid architectures (mixing Convolutions and Attention) offer better priors for simulation tasks?
3.  How do state-space and linear attention models handle the "inverse" problems of Induction and Abduction?

### 2. The ECA-Zero Benchmark

We introduce **ECA-Zero**, a dataset of 333,333 training examples and 3,333 disjoint test examples. Unlike standard sequence prediction tasks, ECA-Zero frames the problem through three distinct modes of reasoning, each accompanied by a specific algorithmic "Chain-of-Thought" (CoT) trace.

#### 2.1 Task Types
*   **Deduction (Forward Simulation):** Given a rule and a start state, predict the future state. The CoT trace forces the model to perform **Explicit Convolution**, scanning the state window-by-window. This tests the model's ability to learn and apply a local lookup table.
*   **Induction (Rule Discovery):** Given start/end states and a complexity class hint, identify the rule. The CoT trace simulates a **Stochastic Search**, where the model proposes hypotheses, simulates them, and verifies against the target.
*   **Abduction (Reverse Engineering):** Given an end state and a rule, find the start state. The CoT trace utilizes **Likelihood Propagation**, calculating local priors for bit transitions and performing greedy constraint satisfaction.

#### 2.2 Complexity Stratification
The dataset is stratified by Wolfram Complexity Classes to ensure models are tested against varying levels of entropy:
*   **Class 1:** Converges to uniformity (Order).
*   **Class 2:** Periodic/Repetitive structures.
*   **Class 3:** Chaotic/Aperiodic (High Entropy).
*   **Class 4:** Complex computation/interaction (The "Edge of Chaos").

### 3. Architectures Under Evaluation

We compare the standard Llama-style Transformer against a suite of efficient and hybrid architectures detailed in our technical appendix.

1.  **Standard Transformer (Llama):** The baseline, utilizing RMSNorm, SwiGLU, and Rotary Embeddings (RoPE).
2.  **Liquid Foundation Model 2 (LFM2):** A hybrid architecture that interleaves standard Grouped Query Attention (GQA) with **Gated Short Convolutions**. This architecture hypothesizes that short convolutions capture local dependencies more efficiently than global attention.
3.  **Multi-Head Latent Attention (MLA):** Derived from DeepSeek-V2, this architecture compresses KV-caches into latent vectors, testing if compression affects reasoning fidelity.
4.  **Kimi Delta Attention (KDA):** A linear attention mechanism that uses a recurrent state update rule ($\mathbf{S}_t$) rather than a historical cache, effectively an RNN with a "Delta Rule" for memory updates.
5.  **PaTH-FoX:** Replaces RoPE with a dynamic "Householder" matrix accumulation and an explicit "Forgetting" layer ($f_t$), allowing data-dependent history decay.
6.  **MesaNet:** Replaces self-attention with an implicit online least-squares optimization problem, maintaining a fixed-size history.
7.  **Native Sparse Attention (NSA) & Mixture-of-Memories (MoM):** (Experiments ongoing).

### 4. Preliminary Results & Emerging Patterns

| Arch | Weights | Deduction: Class 1 | Deduction: Class 2 | Deduction: Class 3 | Deduction: Class 4 | Deduction: Total | Induction: Class 1 | Induction: Class 2 | Induction: Class 3 | Induction: Class 4 | Induction: Total | Abduction: Class 1 | Abduction: Class 2 | Abduction: Class 3 | Abduction: Class 4 | Abduction: Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Llama Huggingface (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/transformer-baseline-1-epoch) | 43.4% | 28.3% | 26.0% | 28.0% | 28.9% | 72.6% | 52.0% | 27.3% | 29.2% | 37.2% | 38.3% | 40.5% | 13.1% | 17.1% | 20.9% |
| Llama Huggingface (3ep) | [Weights](https://huggingface.co/ChavyvAkvar/transformer-baseline-3-epoch) | 31.0% | 27.4% | 35.9% | 27.6% | 30.8% | 30.1% | 54.6% | 60.9% | 49.1% | 52.5% | 14.9% | 8.1% | 12.4% | 13.4% | 12.1% |
| LFM2 Huggingface (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/lfm2-baseline-1-epoch) | 55.8% | 43.8% | 47.6% | 39.5% | 44.8% | 95.6% | 18.9% | 16.2% | 15.6% | 24.2% | 14.9% | 4.3% | 4.6% | 6.5% | 5.8% |
| **LFM2 Huggingface (3ep)** | [Weights](https://huggingface.co/ChavyvAkvar/lfm2-baseline-3-epoch) | **72.6%** | **66.8%** | **81.6%** | **72.9%** | **74.8%** | **92.0%** | **65.6%** | **74.9%** | **71.8%** | **73.6%** | **23.4%** | **44.3%** | **18.3%** | **19.9%** | **23.9%** |
| Transformer FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/transformer-fla-baseline-1-epoch) | 67.3% | 18.1% | 11.4% | 12.7% | 18.6% | 61.1% | 11.5% | 17.4% | 7.3% | 16.9% | 34.0% | 34.1% | 20.1% | 28.4% | 26.5% |
| Transformer FLA (3ep) | | | | | | | | | | | | | | | | |
| Multi Head Latent Attn FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/mla-baseline-1-epoch) | 82.3% | 36.3% | 28.4% | 32.9% | 36.8% | 92.0% | 15.0% | 2.7% | 0.2% | 12.9% | 25.5% | 31.9% | 9.8% | 14.2% | 16.3% |
| Multi Head Latent Attn FLA (3ep) | | | | | | | | | | | | | | | | |
| Kimi Delta Attn FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/kda-baseline-1-epoch) | 50.4% | 44.2% | 36.7% | 38.8% | 40.2% | 37.2% | 8.8% | 1.9% | 0.2% | 6.1% | 19.1% | 23.8% | 5.7% | 4.4% | 9.1% |
| Kimi Delta Attn FLA (3ep) | | | | | | | | | | | | | | | | |
| *PaTH FLA (1ep)* | [Weights](https://huggingface.co/ChavyvAkvar/pathfox-baseline-1-epoch) | 53.1% | 20.4% | 7.8% | 5.4% | 13.8% | *100.0%* | *89.0%* | *96.1%* | *94.2%* | *94.4%* | *59.6%* | *62.7%* | *28.1%* | *34.1%* | *38.2%* |
| PaTH FLA (3ep) | | | | | | | | | | | | | | | | |
| MesaNet FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/mesanet-baseline-1-epoch) | 14.2% | 11.5% | 11.9% | 12.4% | 12.2% | 0.0% | 1.8% | 1.0% | 0.2% | 0.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MesaNet FLA (3ep) | | | | | | | | | | | | | | | | |
| Native Sparse Attn FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/nsa-baseline-1-epoch) | | | | | | | | | | | | | | | | |
| Native Sparse Attn FLA (3ep) | | | | | | | | | | | | | | | | |
| Mixture of Memories FLA (1ep) | [Weights](https://huggingface.co/ChavyvAkvar/mom-baseline-1-epoch) | | | | | | | | | | | | | | | | |
| Mixture of Memories FLA (3ep) | | | | | | | | | | | | | | | | |

*Note: The results discussed below represent intermediate checkpoints (1-epoch and partial 3-epoch runs). Comprehensive training is ongoing.*

#### 4.1 The "Local" Bias in Deduction
In the Deduction task (forward simulation), the **LFM2** architecture (3-epoch) currently demonstrates superior performance (74.8% accuracy) compared to the standard Transformer (30.8%) and other linear variants.
*   **Hypothesis:** ECA rules are inherently local (a cell depends only on its immediate neighbors). LFM2's **Gated Short Convolutions** appear to provide a strong inductive bias for this locality, allowing it to learn the "physics" of the cellular automata faster than the global mixing of standard Attention.

#### 4.2 The "Forgetting" Advantage in Induction
The most striking anomaly in our preliminary data is the performance of **PaTH-FoX** on the Induction task. Even at 1 epoch, PaTH achieves **94.4%** accuracy, significantly outperforming LFM2 (24.2%) and Llama (37.2%).
*   **Observation:** The Induction task involves a "search" trace where the model must simulate a hypothesis, fail, and then try a new hypothesis.
*   **Hypothesis:** PaTH-FoX includes an explicit **forgetting gate** ($f_t$). We hypothesize that this mechanism allows the model to effectively "flush" its hidden state between failed hypothesis traces in the Chain-of-Thought, preventing the noise of a wrong guess from corrupting the next attempt. Standard Transformers, which attend to the entire history, may struggle to ignore the "distractor" traces generated during the search process.

#### 4.3 The Difficulty of Chaos (Class 3 & 4)
Across all architectures, performance drops significantly for Wolfram Class 3 (Chaotic) and Class 4 (Complex) rules, particularly in Abduction (Reverse Engineering).
*   **Abduction:** This is the hardest task. PaTH (38.2%) and LFM2 (23.9%) are currently the only models showing non-trivial traction.
*   **MesaNet:** In our current configuration, MesaNet is struggling (0.8% Induction, 0.0% Abduction). This suggests that the implicit optimization objective may be unstable for discrete, high-frequency state transitions found in ECA, or requires specific hyperparameter tuning distinct from natural language.

### 5. Ongoing Work

The landscape of sequence modeling is diversifying rapidly. Our initial data suggests that "reasoning" is not a monolithic capability; rather, specific architectural choices (like local convolutions for simulation or explicit forgetting for multi-step search) create distinct advantages for different types of algorithmic tasks.

We are currently completing the training runs for:
1.  **Native Sparse Attention (NSA):** To see if hierarchical sparsity can handle Class 4 complexity.
2.  **Mixture-of-Memories (MoM):** To test if routed memory states can separate the distinct rules of ECA better than a single shared memory.
3.  **3-Epoch Convergence:** Completing the training for all architectures to ensure fair comparison at convergence.

Final conclusions regarding the optimal architecture for algorithmic reasoning will be drawn once the full suite of experiments is complete.