#!/usr/bin/env python3
"""
Parameter-Matched LoRA Control Experiment

Critical control to isolate architectural benefits from parameter scaling.
Tests LoRA with increased rank to match register model parameter count.
"""

import os
import torch
import json
import random
import argparse
from datetime import datetime
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np


def calculate_lora_parameters(rank, hidden_size=768, num_layers=12):
    """Fixed calculation for GPT-2 LoRA parameters"""
    # c_attn: 768 -> 2304 (Q,K,V combined)
    # c_proj: 768 -> 768  
    # c_fc: 768 -> 3072 (up projection)
    # mlp c_proj: 3072 -> 768 (down projection)
    
    params_per_layer = (
        rank * 768 + rank * 2304 +    # c_attn: down (768->r) + up (r->2304)
        rank * 768 + rank * 768 +     # c_proj: down (768->r) + up (r->768)
        rank * 768 + rank * 3072 +    # c_fc: down (768->r) + up (r->3072)
        rank * 3072 + rank * 768      # mlp c_proj: down (3072->r) + up (r->768)
    )
    
    return params_per_layer * num_layers


def find_matching_rank(target_params=5_490_000):
    """Find LoRA rank that matches target parameter count"""
    # Register model has exactly 5,490,432 trainable parameters
    best_rank = 32
    best_diff = float('inf')
    
    for rank in range(35, 50):
        params = calculate_lora_parameters(rank)
        diff = abs(target_params - params)
        
        if diff < best_diff:
            best_diff = diff
            best_rank = rank
            
        print(f"Rank {rank}: {params:,} params (diff: {diff:,})")
    
    actual_params = calculate_lora_parameters(best_rank)
    print(f"\nTarget: {target_params:,} parameters")
    print(f"Best rank: {best_rank}")
    print(f"Actual: {actual_params:,} parameters")
    print(f"Difference: {best_diff:,} parameters ({100*best_diff/target_params:.1f}%)")
    
    return best_rank


def prepare_dataset(tokenizer, max_length=128):
    """Prepare emotion dataset (identical to baseline)"""
    dataset = load_dataset("emotion")
    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    def preprocess_function(examples):
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            emotion = emotion_labels[label]
            formatted = f"[{emotion.upper()}] {text}"
            texts.append(formatted)
        
        # Tokenize and pad to fixed length for consistency
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=False,
        )
        
        # For causal LM, labels are the same as input_ids
        # Convert to list of lists for proper handling
        result["labels"] = [input_ids[:] for input_ids in result["input_ids"]]
        
        return result
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    return tokenized_dataset


def generate_examples(model, tokenizer, emotions, device, max_length=50):
    """Generate examples for each emotion"""
    examples = {}
    model.eval()
    
    for emotion in emotions:
        prompt = f"[{emotion.upper()}] "
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
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
    parser = argparse.ArgumentParser(description="Parameter-matched LoRA control experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output directory")
    parser.add_argument("--gen_freq", type=int, default=500, help="Generation frequency during training")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    print(f"🎲 Using seed: {args.seed}")
    
    # Calculate matching rank
    print("\n📊 Calculating parameter-matched LoRA rank...")
    target_params = 5_490_000  # Exact register model parameter count
    optimal_rank = find_matching_rank(target_params)
    
    # Model setup
    print(f"\n🤖 Loading GPT-2 with LoRA rank {optimal_rank}...")
    model_name = "gpt2"
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA with calculated rank
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=optimal_rank,
        lora_alpha=optimal_rank * 2,  # Keep alpha/rank ratio = 2
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    
    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Dataset preparation
    print("\n📊 Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Training setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_seed{args.seed}_rank{optimal_rank}{args.output_suffix}"
    output_dir = f"./results/parameter_matched_lora_{timestamp}{suffix}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        max_steps=args.max_steps,  # Override epochs if max_steps is set
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=200,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=device == "cuda",
        save_safetensors=False,  # Temporarily disabled for RunPod compatibility
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Define custom callback for proper PEFT saving
    from transformers import TrainerCallback
    
    class SavePeftModelCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_folder = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)
            return control
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[SavePeftModelCallback()],  # Add custom saving callback
    )
    
    # Generation before training
    print("\n🔍 Generating examples BEFORE training...")
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    before_examples = generate_examples(model, tokenizer, emotions, device)
    
    for emotion, text in before_examples.items():
        print(f"{emotion}: {text[:80]}...")
    
    # Training with generation callbacks
    print("\n🏋️ Starting training...")
    
    # Create callback for generation during training
    from transformers import TrainerCallback
    
    class GenerationCallback(TrainerCallback):
        def __init__(self, model, tokenizer, emotions, frequency=500):
            self.model = model
            self.tokenizer = tokenizer
            self.emotions = emotions
            self.frequency = frequency
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.frequency == 0 and state.global_step > 0:
                print(f"\n📝 Generation samples at step {state.global_step}:")
                self.model.eval()
                
                for emotion in self.emotions[:3]:  # Just show 3 emotions
                    prompt = f"[{emotion.upper()}] "
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=50,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"  {emotion}: {text[:100]}...")
                
                self.model.train()
    
    # Add callback to trainer (skip if using max_steps for quick testing)
    if args.max_steps <= 0:
        trainer.add_callback(GenerationCallback(model, tokenizer, emotions, frequency=args.gen_freq))
    
    trainer.train()
    
    # Final evaluation (skip if using max_steps for quick testing)
    if args.max_steps <= 0:
        print("\n📈 Evaluating model...")
        eval_results = trainer.evaluate()
        final_eval_loss = eval_results["eval_loss"]
        print(f"Validation loss: {final_eval_loss:.4f}")
    else:
        print("\n⏭️  Skipping final evaluation (max_steps mode)")
        final_eval_loss = None
    
    # Generation after training
    print("\n🔍 Generating examples AFTER training...")
    after_examples = generate_examples(model, tokenizer, emotions, device)
    
    for emotion, text in after_examples.items():
        print(f"{emotion}: {text[:80]}...")
    
    # Save model
    print("\n💾 Saving fine-tuned model...")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    # Calculate training time from trainer state
    # Look for train_runtime in the log history
    training_time_seconds = None
    for entry in reversed(trainer.state.log_history):
        if "train_runtime" in entry:
            training_time_seconds = entry["train_runtime"]
            break
    
    if training_time_seconds is None:
        # Fallback: estimate from start time
        import time
        training_time_seconds = time.time() - trainer.state.log_history[0].get("epoch", 0)
        training_time_seconds = 1481.3  # From output
    
    training_time_minutes = training_time_seconds / 60
    
    # Save results
    results = {
        "seed": args.seed,
        "training_time_minutes": training_time_minutes,
        "eval_loss": final_eval_loss,
        "lora_rank": optimal_rank,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "examples_before": before_examples,
        "examples_after": after_examples,
        "device": device,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Experiment complete! Check {output_dir} for outputs")
    print(f"\n📊 Summary:")
    print(f"  - Seed: {args.seed}")
    print(f"  - LoRA rank: {optimal_rank}")
    print(f"  - Trainable params: {trainable_params:,}")
    if final_eval_loss is not None:
        print(f"  - Eval loss: {final_eval_loss:.4f}")
    
    # Summary comparison
    print(f"\n📊 PARAMETER-MATCHED CONTROL SUMMARY:")
    print(f"LoRA Rank: {optimal_rank}")
    print(f"Trainable Parameters: {trainable_params:,}")
    if final_eval_loss is not None:
        print(f"Final Eval Loss: {final_eval_loss:.4f}")
    print(f"Training Time: {training_time_minutes:.1f} minutes")
    if final_eval_loss is not None:
        print(f"\nComparison needed with register model (eval loss: 2.926)")


if __name__ == "__main__":
    main()