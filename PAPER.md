# Parametric Register Banks: A Memory-Augmented Approach to Parameter-Efficient Fine-Tuning

Rafael Nogueira  
Independent Researcher  
https://github.com/khalic-lab/prb-lora-exploration

## Abstract

We introduce Parametric Register Banks (PRB), a memory-augmented architecture for parameter-efficient fine-tuning that enhances conditional text generation. PRB extends Low-Rank Adaptation (LoRA) with learned memory registers accessed through differentiable gating mechanisms. On emotion-conditioned text generation using GPT-2 small, PRB achieves 2.974 validation loss compared to 3.357 for standard LoRA—an 11.4% improvement. A parameter-matched control experiment using LoRA with increased rank (5.4M parameters) achieves 3.378 validation loss, performing worse than the baseline. This suggests that (1) additional LoRA parameters can degrade performance when exceeding task-specific optimal rank, and (2) PRB's improvements stem from architectural innovation rather than parameter count. Training curves indicate convergence by 14k steps, with PRB converging faster and achieving better final performance than baselines.

## 1 Introduction

Parameter-efficient fine-tuning methods like LoRA (Hu et al., 2021) have become essential for adapting large language models. However, these methods rely solely on weight modifications, lacking explicit mechanisms for maintaining task-specific information across sequences. This becomes limiting in conditional generation tasks requiring consistent attributes throughout generation.

We propose Parametric Register Banks (PRB), augmenting LoRA with learnable memory registers accessed through content-based gating. Unlike recent register tokens in vision transformers (Darcet et al., 2024), our registers are persistent parametric memories with learned read/write operations.

Our contributions:
- A memory-augmented architecture for parameter-efficient conditional generation
- Evidence that dedicated memory provides benefits beyond parameter scaling
- Analysis showing LoRA rank saturation, where additional parameters harm performance

## 2 Method

### 2.1 Architecture

PRB extends a LoRA-adapted model with:

**Register Bank**: Learnable parameters $R \in \mathbb{R}^{n \times d}$ where $n=6$ registers (one per emotion) and $d=64$ dimensions.

**Gating Network**: Linear projection $W_g \in \mathbb{R}^{h \times n}$ computing gates over registers:
$$g = \sigma(W_g \cdot \text{mean}(H))$$
where $\sigma$ is the sigmoid function, followed by softmax normalization for weighting.

**Update Network**: Per-register networks $f_i: \mathbb{R}^{h+d} \rightarrow \mathbb{R}^d$ that modify registers based on hidden states.

### 2.2 Training Procedure

During forward passes:
1. Initialize from parameter bank: $r_t = R$ (with optional emotion bias)
2. Compute gates from hidden states: $g = \sigma(W_g \cdot \text{mean}(h))$ where $\sigma$ is sigmoid
3. Update registers: $r'_i = r_i + g_i \cdot \tanh(f_i([\text{mean}(h); r_i]))$
4. Apply softmax for blending: $w = \text{softmax}(g)$
5. Blend: $r_{blend} = \sum_i w_i \cdot r'_i$
6. Inject: $h' = h + \alpha \cdot W_r \cdot r_{blend}$ where $\alpha$ is a learnable parameter initialized to 0.1

### 2.3 Implementation Details

The model operates on text formatted as `[EMOTION] text` where emotions are encoded in the prefix. The architecture includes emotion embeddings $E \in \mathbb{R}^{6 \times d}$ that can optionally bias register initialization when emotion IDs are provided. In practice, the model learns emotional associations through the gating mechanism applied to text representations, allowing it to function without explicit emotion IDs during inference.

## 3 Experiments

### 3.1 Setup

**Dataset**: Hugging Face `emotion` dataset (16k train, 2k validation, 2k test)
- 6 emotions: sadness, joy, love, anger, fear, surprise
- Formatted as `[EMOTION] text` with average length ~30 tokens

**Base Model**: GPT-2 small (124M parameters)

**Configurations**:
1. **Baseline**: LoRA rank 32 (4.7M trainable parameters)
2. **PRB**: LoRA rank 32 + registers (5.1M parameters)  
3. **Control**: LoRA rank 37 (5.4M parameters) - matched to PRB

**Training**: 
- 16,000 steps (8 epochs), batch size 8
- Learning rate 1e-3 with cosine schedule
- AdamW optimizer
- Single NVIDIA RTX A5000 GPU (~4 hours per run)

### 3.2 Results

| Model | Parameters | Val Loss | Test Loss | Steps to < 3.0 |
|-------|------------|----------|-----------|----------------|
| Baseline (r=32) | 4.7M | 3.357 | 3.384 | Never |
| Control (r=37) | 5.4M | 3.378 | 3.401 | Never |
| PRB (r=32+reg) | 5.1M | 2.974 | 2.991 | 10,000 |

**Key Observations**:

1. **PRB Advantage**: 11.4% improvement over baseline, 12.0% over control
2. **Rank Saturation**: Control with 15% more parameters performs worse than baseline
3. **Convergence**: PRB reaches sub-3.0 loss by step 10k; baselines never achieve this
4. **Plateau Analysis**: All models plateau by ~14k steps, suggesting longer training unnecessary

### 3.3 Training Dynamics

PRB demonstrates superior training dynamics:
- Faster initial learning (steeper descent in first 4k steps)
- Better final training loss (2.01 vs 2.31-2.34)
- More stable convergence (less oscillation after 10k steps)
- Clear plateau by 14k steps across all models

The control's degraded performance despite more parameters suggests the emotion task has inherent rank limitations. Additional LoRA dimensions introduce interference rather than capacity.

### 3.4 Qualitative Assessment

Generation samples show PRB maintains emotional consistency better:

**Prompt**: `[FEAR] ive been feeling`

**Baseline**: "intimidated lately especially the times when im alone with my friends and family"  
**PRB**: "pretty anxious about everything that could go wrong with this situation"

**Prompt**: `[JOY] today was`

**Baseline**: "a good day but tomorrow will be better because i have more time"  
**PRB**: "absolutely incredible and i cant wait to share all the amazing moments"

While anecdotal, PRB outputs show stronger emotional alignment with prompts.

## 4 Analysis

### 4.1 Why Registers Help

The register architecture provides three advantages:

1. **Dedicated Memory**: Registers maintain emotional state without interfering with language modeling
2. **Sparse Updates**: Gating ensures only relevant registers update, reducing noise
3. **Compositional Structure**: Each emotion gets dedicated capacity rather than shared parameters

### 4.2 LoRA Rank Saturation

The control experiment reveals an important phenomenon: increasing LoRA rank from 32 to 37 degrades performance (3.378 vs 3.357 loss). This suggests:

- Tasks have intrinsic rank requirements
- Exceeding optimal rank introduces harmful degrees of freedom
- Parameter efficiency isn't just about size but allocation

This finding challenges assumptions about monotonic scaling in parameter-efficient methods.

### 4.3 Limitations

**Experimental Scope**: Results are from single runs on one dataset with one model size. Statistical significance and generalization require additional validation.

**Scale**: GPT-2 small may not reflect behavior at larger scales where capacity constraints differ.

**Task Specificity**: Emotion classification with 6 categories may particularly benefit from register architecture.

**Hyperparameters**: The scale parameter α is learnable (initialized to 0.1), allowing the model to adjust register influence during training. Other design choices like register dimensions are fixed.

## 5 Related Work

**Parameter-Efficient Fine-Tuning**: LoRA (Hu et al., 2021) and variants modify behavior through low-rank updates. MoRA (Jiang et al., 2024) uses square matrices for increased capacity. PRB differs by adding orthogonal memory rather than modifying adaptation strategy.

**Memory in Transformers**: Register tokens (Darcet et al., 2024) add learnable tokens to vision transformers. Memorizing Transformers (Wu et al., 2022) use external memory for long sequences. PRB implements parametric memory with content-based addressing specifically for conditioning.

**Conditional Generation**: Prior work includes persona-based dialogue (Zhang et al., 2018) and controllable generation (Keskar et al., 2019). PRB provides a parameter-efficient approach to conditional generation through dedicated memory.

## 6 Future Work

**Immediate Extensions**:
- Multiple random seeds for statistical validation
- Testing on GPT-2 medium/large and modern models (LLaMA, Mistral)
- Other conditional tasks (style transfer, persona modeling)
- Learned α and temperature parameters

**Research Questions**:
- Does register count scale with task complexity?
- Can registers enable multi-condition control?
- How do registers interact with larger LoRA ranks?
- Do registers help with longer sequence generation?

**Theoretical Analysis**:
- Formal characterization of LoRA rank saturation
- Information-theoretic analysis of register capacity
- Connection to mixture-of-experts and modular architectures

## 7 Conclusion

Parametric Register Banks demonstrate that architectural innovations in memory can improve parameter-efficient fine-tuning beyond naive parameter scaling. The 11-12% improvement over controls, combined with evidence of LoRA rank saturation, suggests that intelligent parameter allocation through dedicated structures outperforms simply adding capacity. While preliminary, these results indicate that memory-augmented parameter-efficient methods merit further investigation.

The unexpected finding that increased LoRA rank degrades performance highlights the importance of architectural design over parameter count. As the field pushes toward more efficient adaptation methods, incorporating explicit memory mechanisms may provide a path to better conditional generation without scaling model size.

Code and implementation details: [github.com/yourusername/prb-lora] (available upon publication)

## References

Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision Transformers Need Registers. *ICLR 2024*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

Jiang, S., et al. (2024). MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning. *arXiv preprint arXiv:2405.12130*.

Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). CTRL: A Conditional Transformer Language Model for Controllable Generation. *arXiv preprint arXiv:1909.05858*.

Wu, Y., Rabe, M. N., Hutchins, D., & Szegedy, C. (2022). Memorizing Transformers. *ICLR 2022*.

Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing Dialogue Agents: I have a dog, do you have pets too? *ACL 2018*.
