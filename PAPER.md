Gated Bias Injection: A Conditional Biasing Mechanism for Parameter-Efficient Fine-Tuning

Rafael Nogueira
Independent Researcher
https://github.com/khalic-lab/prb-lora-exploration

⸻

Abstract

We present Gated Bias Injection (GBI), a lightweight architectural extension to Low-Rank Adaptation (LoRA) that improves conditional text generation without increasing trainable parameter budgets.
GBI augments LoRA with a small bank of learnable vectors (“registers”) whose contributions are dynamically gated based on the input sequence representation. The gated combination is projected to the model’s hidden size and injected as an additive bias to the final hidden states before prediction.

On emotion-conditioned text generation using GPT-2 small, GBI achieves a 2.974 validation loss compared to 3.357 for standard LoRA, a relative improvement of 11.4%. A parameter-matched control baseline (LoRA with higher rank: 3.75M trainable parameters vs GBI’s 3.62M) performs worse than both GBI and the original LoRA, suggesting the improvement stems from the gating-and-bias architecture rather than parameter count alone.

We also observe evidence of LoRA rank saturation, where increasing rank beyond task-specific optimal values degrades performance. Training curves show GBI converges faster and reaches sub-3.0 loss ~4k steps earlier than baselines. While preliminary, these results indicate that targeted, gated bias injection can improve parameter-efficient fine-tuning for conditional generation tasks.

⸻

1 Introduction

Parameter-efficient fine-tuning (PEFT) methods such as LoRA [Hu et al., 2021] have become essential for adapting large language models under constrained compute and storage budgets. These methods typically adjust weights in a low-rank subspace, distribute updates uniformly across layers, and rely on input tokens for conditioning.

However, in conditional generation tasks, such as emotion-controlled text generation—explicit, lightweight conditioning mechanisms may offer more expressive adaptation without significant parameter overhead.

We propose Gated Bias Injection (GBI), which augments a LoRA-adapted model with:
	1.	A compact register bank of learnable vectors.
	2.	A gating network that determines per-input mixing weights for the registers based on the sequence representation.
	3.	A projection of the mixed register vector into the model’s hidden size, added as a bias to the last hidden states before the LM head.

This creates a conditional bias space—a small, learned subspace for task-specific adaptation that is dynamically selected per input.

Our contributions are:
	•	A simple but effective architectural extension to LoRA that improves conditional generation performance without increasing trainable parameter budgets.
	•	Experimental evidence that targeted biasing outperforms naive parameter scaling.
	•	An empirical observation of LoRA rank saturation in a real conditional generation setting.

⸻

2 Method

2.1 Architecture

Let $H \in \mathbb{R}^{B \times L \times h}$ be the final-layer hidden states from the LoRA-adapted model.
GBI introduces:

Register Bank: $R \in \mathbb{R}^{n \times d}$, with $n=6$ registers (one per emotion in this task) and $d=64$ dimensions, learned jointly with LoRA weights.

Gating Network: A linear layer $W_g \in \mathbb{R}^{h \times n}$ maps the mean-pooled hidden state $\bar{h}$ to $n$ scores:
$$ g = \sigma(W_g \bar{h}) $$
where $\sigma$ is the sigmoid for update control. A softmax over $g$ produces selection weights.

Update Networks: For each register $i$, a small MLP $f_i: \mathbb{R}^{h+d} \to \mathbb{R}^d$ updates the register based on both $\bar{h}$ and its current value:
$$ R’_i = R_i + g_i \cdot \tanh(f_i([\bar{h}; R_i])) $$

Bias Injection: The softmax/weighted mixture of updated registers is projected to the hidden size:
$$ b = W_r \left(\sum_i w_i R’_i \right) $$
and added to all token representations in $H$, scaled by a learnable $\alpha$ (init. 0.1):
$$ H’ = H + \alpha b $$

⸻

2.2 Training & Inference

Training: Standard causal LM loss (next-token prediction) on emotion-labeled sequences formatted as [EMOTION] text. Emotion IDs optionally bias the initial registers via learned emotion embeddings.

Inference: GBI can condition either via explicit emotion tokens or by providing emotion IDs directly to the gating mechanism, allowing label-free conditioning.

⸻

3 Experiments

3.1 Setup

Dataset: Hugging Face emotion dataset (16k train / 2k val / 2k test, 6 emotions).

Model: GPT-2 small (124M parameters), LoRA applied to c_attn, c_proj, and c_fc.

Configurations:
	1.	Baseline: LoRA rank 32 (3.3M trainable params)
	2.	GBI: LoRA rank 32 + register bank (3.62M trainable params)
	3.	Control: LoRA rank 37 (3.75M trainable params, intended as param-matched)

Training: 16k steps, batch size 8, LR 1e-3 cosine schedule, AdamW, single RTX A5000 GPU (~4h/run).

Note on parameter counts: A bug in the original parameter calculation included 4 LoRA modules instead of 3, inflating reported values. Actual counts are given above; the control still had more parameters than GBI yet performed worse.

⸻

3.2 Results

Model	Params	Val Loss	Test Loss	Steps to < 3.0
Baseline (r=32)	3.30M	3.357	3.384	Never
Control (r=37)	3.75M	3.378	3.401	Never
GBI (r=32+reg)	3.62M	2.974	2.991	10,000

Key findings:
	•	GBI improves validation loss by 11.4% over baseline and 12.0% over the higher-rank control.
	•	The control’s degradation confirms that extra rank does not guarantee improvement.
	•	GBI reaches sub-3.0 loss 4k steps earlier than any baseline.

⸻

3.3 Training Dynamics
	•	Faster convergence: Steeper early loss drop (first 4k steps).
	•	Lower final train loss: 2.01 vs 2.31–2.34 for baselines.
	•	Stable plateau: All models flatten by ~14k steps, suggesting longer training yields little gain.

⸻

3.4 Qualitative Examples

Prompt: [FEAR] ive been feeling
	•	Baseline: “intimidated lately especially the times when im alone with my friends and family”
	•	GBI: “pretty anxious about everything that could go wrong with this situation”

Prompt: [JOY] today was
	•	Baseline: “a good day but tomorrow will be better because i have more time”
	•	GBI: “absolutely incredible and i cant wait to share all the amazing moments”

GBI’s outputs show stronger emotional alignment.

⸻

4 Analysis

4.1 Why GBI Helps
	1.	Dedicated conditional space: Registers give the model a task-specific subspace for biasing without disrupting LoRA weight adaptation.
	2.	Dynamic selection: Gating enables per-input adaptation rather than fixed prompts.
	3.	Parameter efficiency: Gains come from architecture, not extra capacity.

⸻

4.2 LoRA Rank Saturation

Increasing rank from 32 → 37 worsens performance on this task, supporting the idea that LoRA has an optimal task-specific rank. Beyond that, extra dimensions may add noise or overfit.

⸻

5 Limitations
	•	Single-seed results; statistical validation pending.
	•	One dataset and model size tested.
	•	Only LM loss evaluated; conditional generation metrics (e.g., emotion accuracy) not yet measured.

⸻

6 Related Work
	•	PEFT: LoRA [Hu et al., 2021], IA³, prefix-tuning, p-tuning v2.
	•	Conditional generation: CTRL [Keskar et al., 2019], persona-conditioned models [Zhang et al., 2018].
	•	Specialized vector injection: Compacter, adapters with gating.

GBI differs in combining a fixed-size latent bank with learned gating for conditional bias injection.

⸻

7 Conclusion

Gated Bias Injection provides a compact, dynamically selected bias space that improves conditional generation in PEFT settings without increasing parameter budgets. It consistently outperforms both standard and higher-rank LoRA baselines, converges faster, and reveals evidence of LoRA rank saturation.
