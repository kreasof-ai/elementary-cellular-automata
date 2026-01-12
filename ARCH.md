# Arch

---

## Transformer

This architecture closely mirrors the "Llama" family of models. We assume a Pre-Normalization architecture (Norm applied before attention/FFN) using **RMSNorm**, which is the standard pairing for these components.

### 1. Notation and Dimensions

*   **$T$**: Sequence length.
*   **$d$**: Hidden dimension size ($d_{model}$).
*   **$h$**: Number of attention heads.
*   **$d_k$**: Dimension per head ($d / h$).
*   **$X_l$**: Input tensor to layer $l$ of shape $[T, d]$.

### 2. Helper Functions

#### A. RMSNorm (Root Mean Square Normalization)
Given an input vector $x$, and a learnable gain parameter $\gamma$:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma
$$

#### B. Swish / SiLU Activation
$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

#### C. Rotary Positional Embeddings (RoPE)
Given a vector $x$ at position $m$ (typically a query or key vector per head), we rotate pairs of elements. For a pair of features $(x_1, x_2)$ and a rotation angle $\theta_i = 10000^{-2(i-1)/d_k}$:

$$
\text{RoPE}(x, m) = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_{d_k-1} \\ x_{d_k} \end{pmatrix} \otimes \cos(m\theta) + \begin{pmatrix} -x_2 \\ x_1 \\ \vdots \\ -x_{d_k} \\ x_{d_k-1} \end{pmatrix} \otimes \sin(m\theta)
$$

*(Note: This is applied element-wise to pairs across the head dimension.)*

### 3. The Algorithm (Single Layer)

Input: $X_{l-1}$ (Output from previous layer)

#### Step 1: Attention Block (with RoPE)

1.  **Normalization:**
    $$
    \tilde{X} = \text{RMSNorm}(X_{l-1})
    $$

2.  **Projections (Linear Layers):**
    $$
    Q = \tilde{X} W_Q, \quad K = \tilde{X} W_K, \quad V = \tilde{X} W_V
    $$
    *Where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$*

3.  **Split into Heads:**
    Reshape $Q, K, V$ into shape $[T, h, d_k]$.

4.  **Apply RoPE (to Q and K only):**
    For every position $m \in \{1, \dots, T\}$ and head $j$:
    $$
    Q_{m,j}' = \text{RoPE}(Q_{m,j}, m)
    $$
    $$
    K_{m,j}' = \text{RoPE}(K_{m,j}, m)
    $$

5.  **Scaled Dot-Product Attention:**
    $$
    \text{Score} = \frac{Q' (K')^T}{\sqrt{d_k}} + M
    $$
    *(Where $M$ is the causal mask matrix)*
    $$
    \text{Attn} = \text{Softmax}(\text{Score}) \cdot V
    $$

6.  **Concatenate and Output Projection:**
    Concat heads back to $[T, d]$:
    $$
    O_{attn} = \text{Concat}(\text{Attn}_1, \dots, \text{Attn}_h) W_O
    $$

7.  **Residual Connection:**
    $$
    H_{mid} = X_{l-1} + O_{attn}
    $$

#### Step 2: Feed-Forward Block (SwiGLU)

1.  **Normalization:**
    $$
    \tilde{H} = \text{RMSNorm}(H_{mid})
    $$

2.  **SwiGLU Computation:**
    The SwiGLU layer utilizes 3 matrices: $W_{gate}$, $W_{in}$ (sometimes called $W_{up}$), and $W_{out}$ (sometimes called $W_{down}$). Typically, the hidden dimension of the FFN is $4d$ or $\frac{8}{3}d$.

    $$
    \text{Gate} = \text{SiLU}(\tilde{H} W_{gate})
    $$
    $$
    \text{Value} = \tilde{H} W_{in}
    $$
    $$
    Y = \text{Gate} \odot \text{Value}
    $$

3.  **Output Projection:**
    $$
    O_{ffn} = Y W_{out}
    $$

4.  **Residual Connection:**
    $$
    X_{l} = H_{mid} + O_{ffn}
    $$

### 4. Summary Algorithm Flow

$$
\begin{aligned}
\text{1. } & \tilde{X} = \text{RMSNorm}(X_{l-1}) \\
\text{2. } & Q, K, V = \text{SplitHeads}(\text{Linear}(\tilde{X})) \\
\text{3. } & Q_{rot}, K_{rot} = \text{RoPE}(Q, K) \\
\text{4. } & \text{AttnOut} = \text{Softmax}\left(\frac{Q_{rot}K_{rot}^T}{\sqrt{d_k}} + M\right) V \\
\text{5. } & X_{mid} = X_{l-1} + \text{AttnOut} W_O \\
\text{6. } & \tilde{X}_{mid} = \text{RMSNorm}(X_{mid}) \\
\text{7. } & \text{FFN}_{out} = (\text{SiLU}(\tilde{X}_{mid} W_{gate}) \odot (\tilde{X}_{mid} W_{in})) W_{out} \\
\text{8. } & X_l = X_{mid} + \text{FFN}_{out}
\end{aligned}
$$

---

## Liquid Foundation Model 2 (LFM2)

The core innovation of LFM2 is a **Hybrid Backbone** that interleaves standard **Grouped Query Attention (GQA)** blocks with computationally efficient **Gated Short Convolution** blocks.

### 1. Model Hyperparameters & Definitions

*   **$L$**: Number of layers.
*   **$d$**: Hidden dimension size ($d_{model}$).
*   **$k$**: Kernel size for short convolutions (typically $k=3$).
*   **$N_{heads}$**: Number of Query heads.
*   **$N_{kv}$**: Number of KV heads (where $N_{kv} < N_{heads}$ for GQA).
*   **$\mathcal{I}_{attn}$**: Set of layer indices that use Attention (Global).
*   **$\mathcal{I}_{conv}$**: Set of layer indices that use Convolutions (Local).
    *   *Note: In LFM2, $|\mathcal{I}_{conv}| > |\mathcal{I}_{attn}|$. For example, LFM2-1.2B has 16 layers, where 10 are Conv and 6 are Attention.*

### 2. Helper Functions

#### A. RMSNorm
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma
$$

#### B. Depthwise Causal Convolution 1D
For a sequence input $Y$ and kernel size $k$, applying a depthwise convolution that prevents looking ahead (causal):
$$
\text{Conv1D}_k(Y)_t = \sum_{j=0}^{k-1} w_j \odot Y_{t-j}
$$

#### C. QK-Norm (Query-Key Normalization)
LFM2 applies normalization to Queries and Keys *before* attention to stabilize training.
$$
Q_{norm} = \text{RMSNorm}(Q), \quad K_{norm} = \text{RMSNorm}(K)
$$

### 3. The LFM2 Layer Algorithm

Let $X_{l}$ be the input to layer $l$. The architecture consists of a **Sequence Mixer** (either Conv or Attention) followed by a **Feed-Forward Network (FFN)**.

#### Step 1: Sequence Mixer (Hybrid)

First, normalize the input:
$$
\tilde{X} = \text{RMSNorm}(X_l)
$$

**IF** $l \in \mathcal{I}_{conv}$ (**Gated Short Convolution Block**):
This block replaces attention with a local mixing operator.
1.  **Triple Projection:** Project input into three branches: $B$ (Gate 1), $C$ (Gate 2), and $H$ (Value).
    $$
    [B, C, H] = \tilde{X} W_{in\_conv} \quad \text{where } W_{in\_conv} \in \mathbb{R}^{d \times 3d}
    $$
2.  **First Gating:**
    $$
    Y = B \odot H
    $$
3.  **Short Convolution:** Apply depthwise convolution (kernel size $k=3$) along the sequence dimension.
    $$
    Z = \text{Conv1D}_k(Y)
    $$
4.  **Second Gating:**
    $$
    O_{mix} = C \odot Z
    $$
5.  **Output Projection:**
    $$
    \text{MixOut} = O_{mix} W_{out\_conv}
    $$

**ELSE IF** $l \in \mathcal{I}_{attn}$ (**Grouped Query Attention Block**):
This block provides global context retrieval.
1.  **Projections:**
    $$
    Q = \tilde{X} W_Q, \quad K = \tilde{X} W_K, \quad V = \tilde{X} W_V
    $$
2.  **QK-Norm:** (Specific to LFM2/ViT-22B stability techniques)
    $$
    Q = \text{RMSNorm}(Q), \quad K = \text{RMSNorm}(K)
    $$
3.  **RoPE:** Apply Rotary Embeddings to $Q$ and $K$.
    $$
    Q_{rot} = \text{RoPE}(Q), \quad K_{rot} = \text{RoPE}(K)
    $$
4.  **GQA Attention:** Compute attention with KV-repeating/grouping.
    $$
    \text{Attn} = \text{Softmax}\left(\frac{Q_{rot} K_{rot}^T}{\sqrt{d_{head}}} + M\right) V
    $$
5.  **Output Projection:**
    $$
    \text{MixOut} = \text{Attn} W_{out\_attn}
    $$

**Residual Connection:**
$$
H_{mid} = X_l + \text{MixOut}
$$

#### Step 2: Feed-Forward Block (SwiGLU)

All layers (Dense LFM2) utilize a SwiGLU MLP.

1.  **Normalization:**
    $$
    \tilde{H} = \text{RMSNorm}(H_{mid})
    $$

2.  **SwiGLU Mechanism:**
    Project into Gate ($W_g$) and Value ($W_{up}$) branches, then element-wise multiply.
    $$
    \text{Gate} = \text{SiLU}(\tilde{H} W_g)
    $$
    $$
    \text{Val} = \tilde{H} W_{up}
    $$
    $$
    \text{FFN}_{internal} = \text{Gate} \odot \text{Val}
    $$

3.  **Output Projection:**
    $$
    \text{FFN}_{out} = \text{FFN}_{internal} W_{down}
    $$

4.  **Final Residual:**
    $$
    X_{l+1} = H_{mid} + \text{FFN}_{out}
    $$

### 4. Summary of Unique Features
1.  **Gated Short Conv:** Used in the majority of layers ($\approx 60-70\%$). It has linear complexity $O(T)$ regarding sequence length, making the model faster at prefill/decode than pure Transformers.
2.  **QK-Norm:** The addition of RMSNorm specifically on the Q and K projections inside the attention block.
3.  **Hybrid Layout:** The algorithm explicitly switches logic based on the layer index $l$.

---

## Multi-Head Latent Attention (MLA)

This architecture differs from standard Transformers primarily in how the Attention mechanism handles Key-Value generation and Positional Embeddings to maximize inference efficiency.

### 1. Dimensions and Notation

*   **$x_t$**: Input hidden state at step $t$ (shape: $1 \times d_{model}$).
*   **$n_h$**: Number of attention heads.
*   **$d_h$**: Dimension per head.
*   **$d_c$**: KV compression dimension (latent vector size).
*   **$d'_c$**: Query compression dimension.
*   **$d_h^R$**: Per-head dimension for the decoupled RoPE vectors.

### 2. The Algorithm (Single Layer)

Input: $X_{in}$ (Input tensor)

#### Step 1: Multi-Head Latent Attention (MLA)

**1. Normalization**
$$
\tilde{x} = \text{RMSNorm}(X_{in})
$$

**2. Latent Vector Generation (Compression)**
Instead of projecting directly to heads, we project to compressed latent vectors.
*   **Query Latent:**
    $$
    c_{Q} = \tilde{x} W_{DQ} \quad \in \mathbb{R}^{d'_c}
    $$
*   **KV Latent:**
    $$
    c_{KV} = \tilde{x} W_{DKV} \quad \in \mathbb{R}^{d_c}
    $$
    *(Note: During inference, only $c_{KV}$ is cached, significantly reducing memory usage compared to standard KV caches.)*

**3. Head Generation (Up-Projection & Decoupled RoPE)**
MLA splits the Query and Key into a "Content" part (derived from latent vectors) and a "RoPE" part (derived separately to preserve positional information).

For each head $i \in \{1, \dots, n_h\}$:

*   **Query Generation:**
    $$
    q_{i}^{C} = c_{Q} W_{UQ}^{i} \quad (\text{Content part})
    $$
    $$
    q_{i}^{R} = \text{RoPE}(c_{Q} W_{QR}^{i}) \quad (\text{RoPE part})
    $$
    $$
    \mathbf{q}_i = [q_{i}^{C}, q_{i}^{R}] \quad (\text{Concatenated Query})
    $$

*   **Key Generation:**
    $$
    k_{i}^{C} = c_{KV} W_{UK}^{i} \quad (\text{Content part})
    $$
    $$
    k^{R} = \text{RoPE}(\tilde{x} W_{KR}) \quad (\text{RoPE part, shared across heads or per head})
    $$
    $$
    \mathbf{k}_i = [k_{i}^{C}, k^{R}] \quad (\text{Concatenated Key})
    $$
    *(Note: The RoPE Key $k^R$ is usually generated from the input $\tilde{x}$ directly, not the latent vector, to ensure positional sensitivity.)*

*   **Value Generation:**
    $$
    \mathbf{v}_i = c_{KV} W_{UV}^{i}
    $$

**4. Scaled Dot-Product Attention**
Calculate attention scores using the concatenated Q and K vectors (which include both content and positional info).

$$
A_{i} = \text{Softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_i^T}{\sqrt{d_h + d_h^R}} + M\right) \mathbf{v}_i
$$
*(Where $M$ is the causal mask).*

**5. Output Projection**
$$
O_{attn} = \text{Concat}(A_1, \dots, A_{n_h}) W_O
$$

**6. Residual Connection**
$$
h_{mid} = X_{in} + O_{attn}
$$

#### Step 2: Feed-Forward Network (SwiGLU)

DeepSeek-V2 uses a SwiGLU FFN (often in a Mixture-of-Experts setup, but here is the standard dense formulation for clarity).

**1. Normalization**
$$
\tilde{h} = \text{RMSNorm}(h_{mid})
$$

**2. SwiGLU Gates and Projection**
$$
\text{Gate} = \text{SiLU}(\tilde{h} W_{gate})
$$
$$
\text{Val} = \tilde{h} W_{in}
$$
$$
y = \text{Gate} \odot \text{Val}
$$

**3. Output**
$$
O_{ffn} = y W_{out}
$$

**4. Final Residual**
$$
X_{out} = h_{mid} + O_{ffn}
$$

### Key Difference from Standard Transformer

The mathematical "magic" of MLA lies in the matrix associativity during **inference**.

In a standard transformer, you calculate $Q K^T$. In MLA, the term $q_{i}^{C} (k_{i}^{C})^T$ expands to:
$$
(c_Q W_{UQ}) (c_{KV} W_{UK})^T = c_Q (W_{UQ} W_{UK}^T) c_{KV}^T
$$
Because $W_{UQ}$ and $W_{UK}$ are constant trained matrices, they can be multiplied together into a single matrix. This allows the model to compute attention scores by interacting the **Query Latent** directly with the cached **KV Latent**, without ever needing to materialize the full expanded Key matrix in memory. This is what allows MLA to reduce KV cache by ~93% (as stated in the DeepSeek-V2 abstract).

---

## Kimi Delta Attention (KDA)

Unlike standard Multi-Head Attention (MHA) which stores a history of keys and values (KV-Cache), KDA is a **Linear Attention** mechanism that compresses history into a fixed-size recurrent state matrix $\mathbf{S}_t$.

### 1. Notation and Dimensions

*   **$x_t$**: Input vector at time step $t$ (shape $d$).
*   **$d_k, d_v$**: Head dimensions (typically 128).
*   **$h$**: Number of heads.
*   **$\mathbf{S}_t$**: The recurrent state matrix for a specific head at time $t$ (shape $d_k \times d_v$).
*   **ShortConv**: A lightweight 1D depth-wise convolution (kernel size typically 4) used to capture local dependencies before the linear attention.

### 2. The KDA Layer Algorithm (Recurrent Form)

The KDA mechanism is defined by the **Gated Delta Rule**. For a single head, the process at time step $t$ is:

#### Step 1: Feature Generation
First, input $x_t$ is normalized and projected. KDA applies a short convolution and activation to query/key/values, and utilizes L2 Normalization for stability.

$$
\tilde{x}_t = \text{RMSNorm}(x_t)
$$

**Query, Key, Value:**
$$
\mathbf{q}_t = \text{L2Norm}(\text{SiLU}(\text{ShortConv}(\tilde{x}_t \mathbf{W}_q)))
$$
$$
\mathbf{k}_t = \text{L2Norm}(\text{SiLU}(\text{ShortConv}(\tilde{x}_t \mathbf{W}_k)))
$$
$$
\mathbf{v}_t = \text{SiLU}(\text{ShortConv}(\tilde{x}_t \mathbf{W}_v))
$$

#### Step 2: Gate Generation (Fine-Grained)
KDA utilizes **channel-wise** decay ($\alpha$) rather than head-wise, allowing specific feature dimensions to be forgotten at different rates.

**Decay Gate ($\alpha_t$)**: Parameterized via low-rank projection.
$$
\alpha_t = f_{\text{decay}}(\tilde{x}_t \mathbf{W}_\alpha^{\downarrow} \mathbf{W}_\alpha^{\uparrow}) \in [0, 1]^{d_k}
$$
*(Where $f$ is a sigmoid-like decay function)*

**Update Rate ($\beta_t$)**: Controls how much new information is written.
$$
\beta_t = \text{Sigmoid}(\tilde{x}_t \mathbf{W}_\beta) \in [0, 1]
$$

#### Step 3: State Update (The Gated Delta Rule)
This is the core innovation. The state $\mathbf{S}_{t-1}$ is decayed by $\alpha_t$, then updated via a rank-1 Delta Rule (orthogonalizing $\mathbf{k}$ against the history), and finally the new key-value pair is added.

$$
\mathbf{S}_t = \underbrace{(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)}_{\text{Delta Rule}} \underbrace{(\text{Diag}(\alpha_t) \mathbf{S}_{t-1})}_{\text{Channel Decay}} + \underbrace{\beta_t \mathbf{k}_t \mathbf{v}_t^\top}_{\text{New Memory}}
$$

*Note: In the Kimi implementation, the operations are fused for hardware efficiency, but this is the mathematical logic.*

#### Step 4: Attention Output
The output is computed by querying the updated state.
$$
\mathbf{o}_t^{\text{head}} = \mathbf{S}_t^\top \mathbf{q}_t
$$

#### Step 5: Output Gating and Projection
The outputs from all heads are concatenated. Kimi Linear applies a specialized output gate and GroupNorm/RMSNorm to the attention output to prevent "Attention Sinks."

**Output Gate ($g_t$):**
$$
g_t = \text{Sigmoid}(\tilde{x}_t \mathbf{W}_g^{\downarrow} \mathbf{W}_g^{\uparrow})
$$

**Final Layer Output:**
$$
\mathbf{y}_t = \mathbf{W}_o \left( g_t \odot \text{RMSNorm}_{\text{head}}(\text{Concat}(\mathbf{o}_t^1, \dots, \mathbf{o}_t^h)) \right)
$$

### 3. The Kimi Linear Architecture (Hybrid)

The full Kimi Linear model is not purely KDA; it is a **Hybrid Architecture**.

$$
\text{Layer}_{type} = \begin{cases} 
\text{MLA (Full Attention)}, & \text{if } l \mod 4 = 0 \\
\text{KDA (Linear)}, & \text{otherwise}
\end{cases}
$$

**1. Token Mixing (Attention):**
$$
x_{\text{mid}} = x_{l-1} + \text{Layer}_{\text{Attn}}(\text{RMSNorm}(x_{l-1}))
$$

**2. Channel Mixing (FFN / MoE):**
Kimi Linear typically uses a Mixture-of-Experts (MoE) Feed-Forward Network here, or a standard SwiGLU FFN.
$$
x_{l} = x_{\text{mid}} + \text{MoE}(\text{RMSNorm}(x_{\text{mid}}))
$$

### Summary of Differences from Standard Transformer
1.  **Complexity:** KDA is $O(T)$ (linear) vs Standard $O(T^2)$ (quadratic).
2.  **Memory:** KDA state size is fixed ($d_k \times d_v$), whereas Standard Attention KV-cache grows linearly with sequence length.
3.  **Position Embedding:** KDA **does not** use RoPE (Rotary Embeddings) in the linear layers. The recurrent update mechanism itself acts as a learnable position encoding. The interleaved global (MLA) layers do not use RoPE either, relying on KDA to handle positioning.

---

## PaTH Attention with forgetting layer (PaTH-FoX)

In this architecture, **PaTH replaces RoPE**. While RoPE uses a static rotation based on relative position, PaTH uses a dynamic, data-dependent transformation matrix accumulated along the sequence.

### 1. Notation and Dimensions

*   **$T$**: Sequence length.
*   **$d$**: Hidden dimension size.
*   **$h$**: Number of attention heads.
*   **$d_k$**: Dimension per head ($d / h$).
*   **$x_t$**: Input vector at position $t$.
*   **$w_t$**: Householder vector at position $t$.
*   **$\beta_t$**: Householder scalar at position $t$.
*   **$f_t$**: Forgetting gate scalar at position $t$.

### 2. Helper Functions

#### A. Parameter Generation (Data-Dependent)
For every token $x_t$, we generate the dynamics parameters.
1.  **Forgetting Gate ($f_t$):**
    $$
    f_t = \sigma(u_f^\top x_t + b_f) \in (0, 1)
    $$
    *(Where $\sigma$ is the sigmoid function)*

2.  **Householder Vector ($w_t$) and Scalar ($\beta_t$):**
    $$
    \beta_t = 2 \cdot \sigma(u_\beta^\top x_t + b_\beta) \in (0, 2)
    $$
    To generate $w_t$, the input is passed through a low-rank linear layer, a short 1D convolution (kernel size 3), and L2 normalization:
    $$
    \tilde{w}_t = \text{Conv1D}(\text{Linear}_{low}(X))_t
    $$
    $$
    w_t = \frac{\tilde{w}_t}{\|\tilde{w}_t\|_2}
    $$

3.  **Householder Matrix ($H_t$):**
    $$
    H_t = I - \beta_t w_t w_t^\top
    $$

#### B. Path Accumulation
We define the transition along the path from position $j$ to $i$ (where $j < i$) as the product of Householder matrices and the product of forgetting gates.

**Matrix Path:**
$$
\mathbf{P}_{j \to i} = \prod_{s=j+1}^{i} H_s = H_{j+1} \cdot H_{j+2} \cdots H_i
$$
*(Note: Since matrix multiplication is non-commutative, the order represents the accumulation of state updates from $j$ to $i$.)*

**Forget Path:**
$$
D_{j \to i} = \prod_{s=j+1}^{i} f_s
$$

### 3. The Algorithm (PaTH-FoX Layer)

**Input:** $X_{l-1}$ (Shape $[T, d]$)

#### Step 1: PaTH-FoX Attention Block

1.  **Normalization & Projections:**
    $$
    \tilde{X} = \text{RMSNorm}(X_{l-1})
    $$
    $$
    Q = \tilde{X} W_Q, \quad K = \tilde{X} W_K, \quad V = \tilde{X} W_V
    $$

2.  **Generate Dynamic Parameters:**
    Compute $H_t$ and $f_t$ for all $t \in [1, T]$ based on $\tilde{X}$ (as defined in Helper Functions).

3.  **Compute Unnormalized Attention Weights:**
    Instead of the standard $Q K^T$, PaTH-FoX computes the similarity by transforming the query $q_i$ through the path of matrices between $j$ and $i$, and scaling by the path of forget gates.
    
    For a query at $i$ and key at $j$ ($j \le i$):
    $$
    \text{Score}_{ij} = k_j^\top \left( \mathbf{P}_{j \to i} \right) q_i
    $$
    
    The unnormalized attention weight $A^*_{ij}$ incorporates the forgetting term:
    $$
    A^*_{ij} = D_{j \to i} \cdot \exp\left( \frac{\text{Score}_{ij}}{\sqrt{d_k}} \right)
    $$
    *(Note: If $i=j$, $\mathbf{P} = I$ and $D = 1$. For causal attention, if $j > i$, $A^*_{ij} = 0$.)*

4.  **Normalization & Output:**
    Compute the weighted sum of values:
    $$
    o_i = \frac{\sum_{j=1}^i A^*_{ij} v_j}{\sum_{j=1}^i A^*_{ij}}
    $$
    
    *(This formulation replaces the standard Softmax. The term $D_{j \to i}$ acts as a data-dependent decay on the context history.)*

5.  **Projection & Residual:**
    $$
    O_{attn} = \text{Concat}(o_1 \dots o_T) W_O
    $$
    $$
    H_{mid} = X_{l-1} + O_{attn}
    $$

#### Step 2: Feed-Forward Block (SwiGLU)
*This remains standard.*

1.  **Normalization:**
    $$
    \tilde{H} = \text{RMSNorm}(H_{mid})
    $$

2.  **SwiGLU:**
    $$
    Y = \text{SiLU}(\tilde{H} W_{gate}) \odot (\tilde{H} W_{in})
    $$

3.  **Output & Residual:**
    $$
    X_l = H_{mid} + Y W_{out}
    $$

### 4. Summary of the Attention Formula

The core innovation of **PaTH-FoX** is replacing the positional encoding component of the attention equation:

$$
\text{Attention}(Q, K, V)_i = \frac{\sum_{j \le i} \left( \prod_{s=j+1}^i f_s \right) \exp\left( \frac{k_j^\top \left( \prod_{s=j+1}^i (I - \beta_s w_s w_s^\top) \right) q_i}{\sqrt{d_k}} \right) v_j}{\sum_{j \le i} \left( \prod_{s=j+1}^i f_s \right) \exp\left( \frac{k_j^\top \left( \prod_{s=j+1}^i (I - \beta_s w_s w_s^\top) \right) q_i}{\sqrt{d_k}} \right)}
$$

*   **PaTH ($H_s$):** Tracks complex state changes and relative positions via matrix accumulation.
*   **FoX ($f_s$):** Allows the model to discard irrelevant history dynamically via scalar accumulation.

---

## MesaNet

The MesaNet replaces the standard Self-Attention mechanism with the **Mesa Layer**, which formulates sequence modeling as an online linear least-squares optimization problem solved at every time step.

### 1. Notation and Dimensions

*   **$x_t$**: Input vector at time $t$ (dimension $d$).
*   **$h$**: Number of heads.
*   **$d_h$**: Head dimension.
*   **$n_s$**: State dimension (usually equal to head dimension $d_h$).
*   **$\text{RMSNorm}$**: Root Mean Square Normalization.
*   **$\text{SiLU}$**: Sigmoid Linear Unit activation ($x \cdot \sigma(x)$).
*   **$\text{Conv1D}_4$**: A depth-wise causal convolution with kernel size 4.

### 2. The MesaNet Block Algorithm

Input: $X_{l-1}$ (Output from previous layer). For a specific time step $t$, let the input be $x_t$.

#### Part A: The Mesa Layer (Sequence Mixing)

**1. Input Normalization & Projections**
First, normalize the input and project to query, key, value, and gates.
$$
\tilde{x}_t = \text{RMSNorm}(x_t)
$$

$$
q^{raw}_t = \tilde{x}_t W_Q, \quad k^{raw}_t = \tilde{x}_t W_K, \quad v^{raw}_t = \tilde{x}_t W_V
$$

**2. Local Processing (Convolution & Activation)**
Apply a short causal convolution (window size 4) followed by activation.
$$
q'_t = \text{SiLU}(\text{Conv1D}_4(q^{raw}_{t-3:t}))
$$
$$
k'_t = \text{SiLU}(\text{Conv1D}_4(k^{raw}_{t-3:t}))
$$
$$
v_t = \text{SiLU}(\text{Conv1D}_4(v^{raw}_{t-3:t})) \quad \text{(Note: values are not L2 normalized)}
$$

**3. L2 Normalization (Keys and Queries)**
Normalize keys and queries to lie on the unit sphere.
$$
q_t = \frac{q'_t}{\|q'_t\|_2}, \quad k_t = \frac{k'_t}{\|k'_t\|_2}
$$

**4. Gate Computation**
Compute input ($\beta_t$) and forget ($\gamma_t$) gates using the normalized input $\tilde{x}_t$.
$$
\beta_t = \sigma(\tilde{x}_t W_\beta + b_\beta) \quad \text{(Input strength)}
$$
$$
\gamma_t = \sigma(\tilde{x}_t W_\gamma + b_\gamma) \quad \text{(Forget strength)}
$$

**5. State Update (Per Head)**
Maintain two recurrent matrix states: $H_t$ (Key covariance) and $G_t$ (Value-Key correlation).
$$
H_t = \gamma_t H_{t-1} + \beta_t (k_t k_t^\top)
$$
$$
G_t = \gamma_t G_{t-1} + \beta_t (v_t k_t^\top)
$$

**6. The Mesa Solve (Implicit Optimization)**
Unlike standard attention, MesaNet explicitly solves a linear system to find the optimal "fast weights" $q^*_t$. $\Lambda$ is a learnable diagonal regularization matrix (constrained to be positive via softplus).

$$
q^*_t = (H_t + \Lambda)^{-1} q_t
$$

*Note: In implementation, this inverse is approximated using the **Conjugate Gradient (CG)** method, often truncated to $k$ steps.*

**7. Output Computation**
Query the value state $G_t$ with the optimized query $q^*_t$.
$$
o^{head}_t = G_t q^*_t
$$

**8. Output Projection & Residual**
Concatenate heads, project, normalize, and add residual.
$$
o_{mesa} = \text{Concat}(o^{head}_1, \dots, o^{head}_h) W_O
$$
$$
\tilde{o}_{mesa} = \text{RMSNorm}(o_{mesa})
$$
$$
x_{mid} = x_t + \tilde{o}_{mesa}
$$

#### Part B: Channel Mixing (Gated MLP)

The paper utilizes a standard SwiGLU MLP block.

**1. Normalization**
$$
\tilde{x}_{mid} = \text{RMSNorm}(x_{mid})
$$

**2. Gated MLP**
Project to hidden dimension (typically $4d$ or similar), gate, and project back.
$$
y_{gate} = \text{SiLU}(\tilde{x}_{mid} W_{gate})
$$
$$
y_{in} = \tilde{x}_{mid} W_{in}
$$
$$
y_{out} = (y_{gate} \odot y_{in}) W_{out}
$$

**3. Final Residual**
$$
x_{out} = x_{mid} + y_{out}
$$

### Summary of Differences from Transformers
1.  **State-Based:** It maintains a fixed-size history ($H_t, G_t$) rather than a growing KV cache.
2.  **Implicit Optimization:** The core mechanism is solving $q^* = \text{argmin}_{z} \frac{1}{2} \|v - z^\top k\|^2 + \frac{1}{2} z^\top \Lambda z$ at inference time, represented by the linear system solution $(H_t + \Lambda)^{-1}q_t$.
3.  **Local Convolutions:** It uses short 1D convolutions on $Q, K, V$ before the sequence mixing to capture local smoothness.

---

## Mixture-of-Memories (MoM)

The core innovation of MoM is replacing the single fixed-size memory state of linear attention models with multiple independent memory states, routed via a Top-$k$ mechanism, plus a continuously active shared memory.

### 1. Notation and Dimensions

*   **$x_t \in \mathbb{R}^d$**: Input token representation at step $t$.
*   **$N$**: Total number of routed memory states.
*   **$k$**: Number of active memories per token (Top-$k$).
*   **$M_t^{(i)} \in \mathbb{R}^{d \times d}$**: The recurrent state matrix of the $i$-th memory module.
*   **$M_t^{shared} \in \mathbb{R}^{d \times d}$**: The recurrent state of the shared memory.
*   **$W_g \in \mathbb{R}^{d \times N}$**: Router weights.

### 2. The Algorithm (Single Layer)

#### Step 1: Router Mechanism
The router determines which memory states should process the current token. It computes importance scores and selects the top-$k$ indices.

1.  **Compute Scores:**
    $$ \text{scores}_t = \text{Softmax}(x_t W_g) \in \mathbb{R}^N $$

2.  **Top-k Selection:**
    Select indices $\mathcal{I}_t$ corresponding to the $k$ largest values in $\text{scores}_t$.

3.  **Normalization:**
    Normalize scores for the selected memories to sum to 1.
    $$
    g_t^{(i)} =
    \begin{cases}
    \frac{\text{scores}_t^{(i)}}{\sum_{j \in \mathcal{I}_t} \text{scores}_t^{(j)}} & \text{if } i \in \mathcal{I}_t \\
    0 & \text{otherwise}
    \end{cases}
    $$

#### Step 2: Shared Memory Update
The shared memory is always active and captures global context.

1.  **Projections:**
    $$ k_t^{sh} = x_t W_K^{sh}, \quad v_t^{sh} = x_t W_V^{sh} $$

2.  **State Update:**
    $$ M_t^{shared} = \text{Update}(M_{t-1}^{shared}, k_t^{sh}, v_t^{sh}) $$

#### Step 3: Routed Memories Update
For each active memory index $i \in \mathcal{I}_t$, we compute specific keys/values and update the state. Inactive memories remain unchanged.

1.  **Projections (Sparse Activation):**
    $$ k_t^{(i)} = x_t W_K^{(i)}, \quad v_t^{(i)} = x_t W_V^{(i)} $$

2.  **State Update (Active Memories):**
    $$ M_t^{(i)} = \text{Update}(M_{t-1}^{(i)}, k_t^{(i)}, v_t^{(i)}) $$

3.  **Inactive Memories:**
    $$ M_t^{(j)} = M_{t-1}^{(j)} \quad \forall j \notin \mathcal{I}_t $$

*(Note: The `Update` function depends on the underlying linear RNN used. In the paper's experiments, **Gated DeltaNet** is used, but the framework supports standard Linear Attention updates: $M_t = M_{t-1} + k^T v$.)*

#### Step 4: Memory Mixing
The effective memory state for the current time step is a weighted sum of the active routed memories plus the shared memory.

$$ \tilde{M}_t = M_t^{shared} + \sum_{i \in \mathcal{I}_t} g_t^{(i)} M_t^{(i)} $$

#### Step 5: Output Generation
The output is generated by querying the mixed memory state.

1.  **Query Projection:**
    $$ q_t = x_t W_Q $$

2.  **Retrieval:**
    $$ o_t = q_t \tilde{M}_t $$

3.  **Final Output:**
    $$ y_t = \text{RMSNorm}(o_t) W_O + x_t $$

### Summary of Update Mechanisms
While MoM is a general framework, the paper highlights **Gated DeltaNet** (Yang et al., 2024) as the specific update mechanism used in their best-performing models.

**If using Standard Linear Attention Update:**
$$ \text{Update}(M, k, v) = M + k^T v $$

**If using Gated DeltaNet Update (as per Section 4.1):**
$$ \text{Update}(M, k, v) = (I - k^T k) M + v^T k \quad \text{(Simplified form)} $$
*(Note: Actual Gated DeltaNet implementation involves data-dependent decays $\alpha$ and gate $\beta$, making the update: $M_t = \alpha_t M_{t-1} + \beta_t (v_t - M_{t-1} k_t)^T \otimes k_t$)*

---

## Native Sparse Attention (NSA)

### 1. Overview & Notation

NSA replaces standard Full Attention with a hierarchical mechanism containing three parallel branches:
1.  **Token Compression:** Summarizes blocks of tokens for coarse-grained global context.
2.  **Token Selection:** Uses importance scores from the compressed branch to retrieve specific fine-grained blocks.
3.  **Sliding Window:** Captures local context.

**Inputs:**
*   Query vector at step $t$: $\mathbf{q}_t \in \mathbb{R}^D$
*   Key/Value sequence up to $t$: $\mathbf{K}_{:t}, \mathbf{V}_{:t}$
*   Block size $l$, Stride $d$, Window size $w$, Selection count $n$.

### 2. Branch 1: Compressed Attention (Coarse-Grained)

This branch reduces the sequence length by compressing blocks of keys/values into single representations using a learnable MLP with intra-block positional encoding ($\phi$).

**Step 1: Divide into blocks**
The past Key/Value sequence is divided into blocks of size $l$ with stride $d$.

**Step 2: Compress blocks**
For the $i$-th block (indices $id+1$ to $id+l$), the compressed key $\tilde{\mathbf{k}}^{\text{cmp}}_i$ is computed:
$$
\tilde{\mathbf{k}}^{\text{cmp}}_i = \phi_K(\mathbf{k}_{id+1 : id+l})
$$
$$
\tilde{\mathbf{v}}^{\text{cmp}}_i = \phi_V(\mathbf{v}_{id+1 : id+l})
$$
*Note: $\phi$ is an MLP that maps $\mathbb{R}^{l \times D} \to \mathbb{R}^D$.*

**Step 3: Compute Compressed Attention**
$$
\mathbf{o}^{\text{cmp}}_t = \text{Attn}(\mathbf{q}_t, \tilde{\mathbf{K}}^{\text{cmp}}, \tilde{\mathbf{V}}^{\text{cmp}})
$$

### 3. Branch 2: Selected Attention (Fine-Grained)

This branch selects the most relevant original token blocks based on the attention scores generated in the Compressed branch.

**Step 1: Calculate Block Importance**
The importance score $\mathbf{p}^{\text{cmp}}_t$ for each block is derived from the attention scores of the compressed keys calculated in Branch 1:
$$
\mathbf{p}^{\text{cmp}}_t = \text{Softmax}\left( \frac{\mathbf{q}_t (\tilde{\mathbf{K}}^{\text{cmp}})^T}{\sqrt{D}} \right)
$$

**Step 2: Top-$n$ Block Selection**
Identify the indices $\mathcal{I}_t$ of the top-$n$ blocks with the highest importance scores:
$$
\mathcal{I}_t = \{ i \mid \text{rank}(\mathbf{p}^{\text{cmp}}_t[i]) \le n \}
$$

**Step 3: Gather Fine-Grained Tokens**
Retrieve the original (uncompressed) Key/Value blocks corresponding to indices $\mathcal{I}_t$:
$$
\tilde{\mathbf{K}}^{\text{slc}} = \text{Concat}(\{ \mathbf{k}_{id+1 : id+l} \mid i \in \mathcal{I}_t \})
$$
$$
\tilde{\mathbf{V}}^{\text{slc}} = \text{Concat}(\{ \mathbf{v}_{id+1 : id+l} \mid i \in \mathcal{I}_t \})
$$

**Step 4: Compute Selected Attention**
$$
\mathbf{o}^{\text{slc}}_t = \text{Attn}(\mathbf{q}_t, \tilde{\mathbf{K}}^{\text{slc}}, \tilde{\mathbf{V}}^{\text{slc}})
$$

### 4. Branch 3: Sliding Window Attention (Local)

This branch ensures precise local context is always maintained.

**Step 1: Isolate Local Window**
Extract the most recent $w$ tokens:
$$
\tilde{\mathbf{K}}^{\text{win}} = \mathbf{K}_{t-w:t}, \quad \tilde{\mathbf{V}}^{\text{win}} = \mathbf{V}_{t-w:t}
$$

**Step 2: Compute Window Attention**
$$
\mathbf{o}^{\text{win}}_t = \text{Attn}(\mathbf{q}_t, \tilde{\mathbf{K}}^{\text{win}}, \tilde{\mathbf{V}}^{\text{win}})
$$

### 5. Final Aggregation (Gated Summation)

The outputs of the three branches are combined via a learnable gating mechanism.

**Step 1: Compute Gates**
A scalar gate score $g^c$ is computed for each branch $c \in \{\text{cmp}, \text{slc}, \text{win}\}$ using the input features (or query state) passed through an MLP and Sigmoid:
$$
g^{\text{cmp}}_t, g^{\text{slc}}_t, g^{\text{win}}_t = \sigma(\text{MLP}_{\text{gate}}(\mathbf{x}_t))
$$

**Step 2: Weighted Sum**
The final output $\mathbf{o}^*_t$ is the gated sum of the three attention outputs:
$$
\mathbf{o}^*_t = g^{\text{cmp}}_t \mathbf{o}^{\text{cmp}}_t + g^{\text{slc}}_t \mathbf{o}^{\text{slc}}_t + g^{\text{win}}_t \mathbf{o}^{\text{win}}_t
$$

### Summary Algorithm Flow

$$
\begin{aligned}
& \textbf{1. Compression:} \\
& \quad \tilde{K}^{\text{cmp}}, \tilde{V}^{\text{cmp}} \leftarrow \text{BlockCompress}(K_{:t}, V_{:t}) \\
& \quad \text{Scores}^{\text{cmp}} \leftarrow \mathbf{q}_t (\tilde{K}^{\text{cmp}})^T / \sqrt{D} \\
& \quad \mathbf{o}^{\text{cmp}} \leftarrow \text{Softmax}(\text{Scores}^{\text{cmp}}) \tilde{V}^{\text{cmp}} \\
\\
& \textbf{2. Selection:} \\
& \quad \mathcal{I}_{\text{top-}n} \leftarrow \text{TopKIndices}(\text{Scores}^{\text{cmp}}) \\
& \quad \tilde{K}^{\text{slc}}, \tilde{V}^{\text{slc}} \leftarrow \text{GatherBlocks}(K_{:t}, V_{:t}, \mathcal{I}_{\text{top-}n}) \\
& \quad \mathbf{o}^{\text{slc}} \leftarrow \text{Attn}(\mathbf{q}_t, \tilde{K}^{\text{slc}}, \tilde{V}^{\text{slc}}) \\
\\
& \textbf{3. Sliding Window:} \\
& \quad \tilde{K}^{\text{win}}, \tilde{V}^{\text{win}} \leftarrow \text{GetLast}(K_{:t}, V_{:t}, w) \\
& \quad \mathbf{o}^{\text{win}} \leftarrow \text{Attn}(\mathbf{q}_t, \tilde{K}^{\text{win}}, \tilde{V}^{\text{win}}) \\
\\
& \textbf{4. Output:} \\
& \quad \mathbf{o}^*_t = \sum_{c \in \{\text{cmp, slc, win}\}} \text{Gate}_c(\mathbf{x}) \cdot \mathbf{o}^c
\end{aligned}
$$

---