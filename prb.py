"""
LoRA + Register Banks for emotion-conditioned text generation.
Comparing against baseline LoRA to see if persistent memory helps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
import time
import os
import sys
import json
import random
import argparse
from datetime import datetime

from register_bank_unsupervised import RegisterBankTransformerUnsupervised


class LoRAWithRegisters(nn.Module):
    """Combine LoRA-adapted GPT-2 with register banks"""
    
    def __init__(self, model_name="gpt2", n_registers=6, register_dim=64):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,  # Same as baseline
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"],
        )
        self.base_model = get_peft_model(self.base_model, peft_config)
        
        # Register components
        self.n_registers = n_registers
        self.register_dim = register_dim
        self.hidden_size = self.base_model.config.hidden_size
        
        # One register per emotion
        self.register_bank = nn.Parameter(torch.zeros(1, n_registers, register_dim))
        nn.init.normal_(self.register_bank, std=0.02)
        
        # Register controllers
        self.register_gate = nn.Linear(self.hidden_size, n_registers)
        self.register_update = nn.ModuleList([
            nn.Linear(self.hidden_size + register_dim, register_dim) 
            for _ in range(n_registers)
        ])
        
        # Inject registers into sequence
        self.register_proj = nn.Linear(register_dim, self.hidden_size)
        
        # Emotion embedding to select register
        self.emotion_embeddings = nn.Embedding(6, register_dim)
        
        # Learnable register contribution scale
        self.register_scale = nn.Parameter(torch.tensor(0.1))
        
        
    def forward(self, input_ids, attention_mask=None, labels=None, emotion_ids=None):
        B, L = input_ids.shape
        
        # Get base model outputs (includes hidden states)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Initialize registers based on emotion
        if emotion_ids is not None:
            # Use emotion-specific initialization
            emotion_bias = self.emotion_embeddings(emotion_ids)  # [B, register_dim]
            # Broadcast emotion bias to all registers
            emotion_bias = emotion_bias.unsqueeze(1)  # [B, 1, register_dim]
            registers = self.register_bank.expand(B, -1, -1) + 0.1 * emotion_bias
        else:
            registers = self.register_bank.expand(B, -1, -1)
        
        # Update registers based on sequence
        seq_mean = hidden_states.mean(dim=1)  # [B, hidden_size]
        
        # Gated update
        gate = torch.sigmoid(self.register_gate(seq_mean))
        
        new_registers = []
        for i in range(self.n_registers):
            combined = torch.cat([seq_mean, registers[:, i, :]], dim=-1)
            update = self.register_update[i](combined)
            new_reg = registers[:, i, :] + gate[:, i:i+1] * torch.tanh(update)
            new_registers.append(new_reg)
        
        registers = torch.stack(new_registers, dim=1)
        
        # Inject register information back
        # Use the most relevant register based on gating
        register_weights = F.softmax(gate, dim=-1)  # [B, n_registers]
        weighted_registers = (registers * register_weights.unsqueeze(-1)).sum(dim=1)  # [B, register_dim]
        register_info = self.register_proj(weighted_registers).unsqueeze(1)  # [B, 1, hidden_size]
        
        # Track analytics if enabled
        if hasattr(self, 'register_analytics'):
            gate_entropy = -torch.sum(register_weights * torch.log(register_weights + 1e-8), dim=-1).mean()
            
            # Compute update magnitudes
            update_mags = []
            for i in range(self.n_registers):
                orig_reg = self.register_bank[0, i, :].expand_as(registers[:, i, :])
                update_mag = torch.norm(registers[:, i, :] - orig_reg, dim=-1).mean()
                update_mags.append(update_mag)
            
            self.register_analytics.update(
                gate_weights=register_weights.detach(),
                gate_entropy=gate_entropy.detach(),
                dominant_registers=torch.argmax(register_weights, dim=-1).detach(),
                register_values=registers.detach(),
                update_magnitudes=torch.stack(update_mags).detach()
            )
        
        # Add register info to hidden states with learnable scale
        enhanced_hidden = hidden_states + self.register_scale * register_info
        
        # Get final logits
        logits = self.base_model.lm_head(enhanced_hidden)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}
    
    def generate(self, input_ids=None, attention_mask=None, emotion_ids=None, use_registers=True, **kwargs):
        """Generate text with optional register-aware forward pass
        
        Args:
            use_registers: If True, use register-enhanced generation. If False, use base model only.
            emotion_ids: Emotion IDs for register initialization (only used if use_registers=True)
        """
        
        if not use_registers:
            # Simple mode: just use base model without registers
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Register-aware generation mode
        # Store emotion_ids for use in forward
        self._generation_emotion_ids = emotion_ids
        
        # Create a wrapper for forward that includes emotion_ids
        original_forward = self.forward
        
        def wrapped_forward(input_ids, attention_mask=None, **forward_kwargs):
            # Remove keys that our forward doesn't handle
            forward_kwargs.pop('past_key_values', None)
            forward_kwargs.pop('use_cache', None)
            forward_kwargs.pop('position_ids', None)
            forward_kwargs.pop('token_type_ids', None)
            forward_kwargs.pop('head_mask', None)
            forward_kwargs.pop('inputs_embeds', None)
            forward_kwargs.pop('output_attentions', None)
            forward_kwargs.pop('output_hidden_states', None)
            forward_kwargs.pop('return_dict', None)
            
            # Call our forward with emotion_ids
            outputs = original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                emotion_ids=self._generation_emotion_ids,
                **forward_kwargs
            )
            
            # Convert dict to expected output format
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=outputs['logits'],
                past_key_values=None
            )
        
        # Temporarily replace forward
        self.forward = wrapped_forward
        
        try:
            # Generate with our wrapped forward
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        finally:
            # Restore original forward
            self.forward = original_forward
            # Clean up
            if hasattr(self, '_generation_emotion_ids'):
                delattr(self, '_generation_emotion_ids')
        
        return outputs
    
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """Save the model properly, handling both PEFT and register components."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save PEFT adapter
        self.base_model.save_pretrained(save_directory, safe_serialization=safe_serialization)
        
        # Save register components
        register_state = {
            'register_bank': self.register_bank,
            'register_gate': self.register_gate.state_dict(),
            'register_update': {i: m.state_dict() for i, m in enumerate(self.register_update)},
            'register_proj': self.register_proj.state_dict(),
            'emotion_embeddings': self.emotion_embeddings.state_dict(),
            'config': {
                'n_registers': self.n_registers,
                'register_dim': self.register_dim,
                'hidden_size': self.hidden_size
            }
        }
        
        torch.save(register_state, os.path.join(save_directory, "register_components.pt"))
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, device: str = "cpu"):
        """Load a saved model with registers."""
        # Load register components to get config
        register_state = torch.load(
            os.path.join(pretrained_path, "register_components.pt"),
            map_location=device
        )
        
        # Create model with saved config
        model = cls(
            n_registers=register_state['config']['n_registers'],
            register_dim=register_state['config']['register_dim']
        )
        
        # Load PEFT adapter
        from peft import PeftModel
        # First load the base model without PEFT
        base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        # Then load the PEFT adapter
        model.base_model = PeftModel.from_pretrained(
            base_model,
            pretrained_path
        )
        
        # Load register components
        model.register_bank.data = register_state['register_bank']
        model.register_gate.load_state_dict(register_state['register_gate'])
        for i, state_dict in register_state['register_update'].items():
            model.register_update[i].load_state_dict(state_dict)
        model.register_proj.load_state_dict(register_state['register_proj'])
        model.emotion_embeddings.load_state_dict(register_state['emotion_embeddings'])
        
        # Update config
        model.hidden_size = register_state['config']['hidden_size']
        
        return model.to(device)


def prepare_dataset_with_emotions(tokenizer, max_length=128):
    """Prepare dataset with emotion IDs for register selection"""
    
    dataset = load_dataset("emotion")
    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    def preprocess_function(examples):
        texts = []
        emotion_ids = []
        
        for text, label in zip(examples["text"], examples["label"]):
            emotion = emotion_labels[label]
            formatted = f"[{emotion.upper()}] {text}"
            texts.append(formatted)
            emotion_ids.append(label)
        
        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        model_inputs["emotion_ids"] = emotion_ids
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset


class RegisterDataCollator(DataCollatorForLanguageModeling):
    """Custom collator that preserves emotion_ids"""
    
    def __call__(self, features):
        # Extract emotion_ids before standard processing
        emotion_ids = [f.pop("emotion_ids", 0) for f in features]
        
        # Use parent class for standard processing
        batch = super().__call__(features)
        
        # Add emotion_ids back
        batch["emotion_ids"] = torch.tensor(emotion_ids)
        
        return batch


def generate_examples(model, tokenizer, emotions, device, max_length=50):
    """Generate examples for each emotion"""
    model.eval()
    examples = {}
    
    emotion_to_id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
    
    for emotion in emotions:
        prompt = f"[{emotion.upper()}] "
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Add emotion ID
        emotion_id = torch.tensor([emotion_to_id[emotion]]).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                emotion_ids=emotion_id,
                use_registers=True  # Use register-aware generation
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        examples[emotion] = generated_text
    
    return examples


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA + Registers emotion experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output directory")
    parser.add_argument("--gen_freq", type=int, default=500, help="Generation frequency during training")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("🚀 LoRA + Registers Emotion Experiment")
    print("=" * 50)
    print(f"🎲 Using seed: {args.seed}")
    
    # Device detection with detailed info
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print(f"Device: CPU (Warning: This will be very slow!)")
    
    # Environment info
    import platform
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n📚 Creating model with LoRA + Registers...")
    model = LoRAWithRegisters("gpt2", n_registers=6, register_dim=64)
    model = model.to(device)
    
    # Print parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Prepare dataset
    print("\n📊 Preparing dataset with emotion IDs...")
    dataset = prepare_dataset_with_emotions(tokenizer)
    print(f"Training samples: {len(dataset['train'])}")
    
    # Use same batch size as baseline for consistency
    train_batch_size = 8
    eval_batch_size = 8
    print("Using batch size 8 for consistency with baseline")
    
    # Training arguments (same as baseline for fair comparison)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_seed{args.seed}{args.output_suffix}"
    output_dir = f"./results/lora_emotion_registers_{timestamp}{suffix}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        max_steps=args.max_steps,  # Override epochs if max_steps is set
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=200,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,  # Only keep 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=device == "cuda",  # Use mixed precision on CUDA
        save_safetensors=False,  # Temporarily disabled for RunPod compatibility
    )
    
    # Custom data collator
    data_collator = RegisterDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Import custom callback for proper PEFT saving and custom trainer
    # Define custom callback for proper PEFT saving
    from transformers import TrainerCallback
    
    class SavePeftModelCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_folder = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)
            return control
    # Use standard Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SavePeftModelCallback()],  # Add custom saving callback
    )
    
    # Generate before training
    print("\n🔍 Generating examples BEFORE training...")
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    before_examples = generate_examples(model, tokenizer, emotions, device)
    for emotion, text in before_examples.items():
        print(f"\n{emotion}: {text[:100]}...")
    
    # Train with generation callbacks
    print("\n🏋️ Starting training...")
    
    # Create callback for generation during training
    from transformers import TrainerCallback
    
    class GenerationCallback(TrainerCallback):
        def __init__(self, model, tokenizer, emotions, frequency=500):
            self.model = model
            self.tokenizer = tokenizer
            self.emotions = emotions
            self.frequency = frequency
            self.emotion_to_id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.frequency == 0 and state.global_step > 0:
                print(f"\n📝 Generation samples at step {state.global_step}:")
                self.model.eval()
                
                # Test both with and without emotion_id for registers
                for emotion in self.emotions[:3]:  # Just show 3 emotions
                    prompt = f"[{emotion.upper()}] "
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # With emotion_id
                    emotion_id = torch.tensor([self.emotion_to_id[emotion]]).to(self.model.device)
                    with torch.no_grad():
                        outputs_with_id = self.model.generate(
                            **inputs,
                            max_length=50,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            emotion_ids=emotion_id,
                            use_registers=True
                        )
                    
                    # Without emotion_id (for comparison)
                    with torch.no_grad():
                        outputs_no_id = self.model.generate(
                            **inputs,
                            max_length=50,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_registers=False  # Compare against no registers
                        )
                    
                    text_with_id = self.tokenizer.decode(outputs_with_id[0], skip_special_tokens=True)
                    text_no_id = self.tokenizer.decode(outputs_no_id[0], skip_special_tokens=True)
                    
                    print(f"  {emotion} (+reg): {text_with_id[:80]}...")
                    print(f"  {emotion} (-reg): {text_no_id[:80]}...")
                
                self.model.train()
    
    # Add callback to trainer (skip if using max_steps for quick testing)
    if args.max_steps <= 0:
        trainer.add_callback(GenerationCallback(model, tokenizer, emotions, frequency=args.gen_freq))
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate (skip if using max_steps for quick testing)
    if args.max_steps <= 0:
        eval_results = trainer.evaluate()
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    else:
        print("\n⏭️  Skipping final evaluation (max_steps mode)")
        eval_results = {"eval_loss": None}
    
    # Generate after training
    print("\n🔍 Generating examples AFTER training...")
    after_examples = generate_examples(model, tokenizer, emotions, device)
    for emotion, text in after_examples.items():
        print(f"\n{emotion}: {text[:100]}...")
    
    # Save
    print("\n💾 Saving model...")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    # Save results for comparison
    results = {
        "seed": args.seed,
        "training_time_minutes": training_time/60,
        "eval_loss": eval_results['eval_loss'],
        "examples_before": before_examples,
        "examples_after": after_examples,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "register_config": {
            "n_registers": 6,
            "register_dim": 64,
            "emotion_aware": True
        },
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Experiment complete! Check {output_dir} for outputs")
    print(f"\n📊 Summary:")
    print(f"  - Seed: {args.seed}")
    print(f"  - Trainable params: {trainable_params:,}")
    if eval_results['eval_loss'] is not None:
        print(f"  - Eval loss: {eval_results['eval_loss']:.4f}")
        print(f"\n📊 Compare with baseline:")
        print(f"   Baseline eval loss: 3.358")
        print(f"   Registers eval loss: {eval_results['eval_loss']:.4f}")
    else:
        print(f"  - Eval loss: Skipped (max_steps mode)")


if __name__ == "__main__":
    main()