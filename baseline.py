"""
Simple LoRA fine-tuning experiment using the emotion dataset.
This establishes a baseline before adding register banks.
"""

import torch
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
import evaluate
import numpy as np
import time
import os
import json
import random
import argparse
from datetime import datetime


def prepare_dataset(tokenizer, max_length=128):
    """Load and prepare the emotion dataset for causal LM"""
    
    # Load emotion dataset
    dataset = load_dataset("emotion")
    
    # Emotion labels
    emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    
    def preprocess_function(examples):
        """Convert to format: '[EMOTION] text'"""
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            emotion = emotion_labels[label]
            # Format: [EMOTION] text
            formatted = f"[{emotion.upper()}] {text}"
            texts.append(formatted)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Process datasets
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset


def setup_lora_model(model_name="gpt2", lora_rank=32):  # Changed to full GPT-2
    """Setup model with LoRA configuration"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    
    # LoRA configuration with configurable rank
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,  # Configurable LoRA rank
        lora_alpha=lora_rank * 2,  # Alpha = 2*r is a good rule of thumb
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],  # Target more layers
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def compute_metrics(eval_pred, tokenizer):
    """Compute perplexity for evaluation"""
    predictions, labels = eval_pred
    
    # Shift for causal LM
    predictions = predictions[:, :-1, :]
    labels = labels[:, 1:]
    
    # Calculate perplexity
    loss = F.cross_entropy(
        predictions.reshape(-1, predictions.shape[-1]),
        labels.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )
    
    perplexity = torch.exp(loss)
    
    return {"perplexity": perplexity.item()}


def generate_examples(model, tokenizer, emotions, prompt_template="[{}] ", max_length=50):
    """Generate example texts for each emotion"""
    model.eval()
    examples = {}
    device = next(model.parameters()).device
    
    for emotion in emotions:
        prompt = prompt_template.format(emotion.upper())
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
    parser = argparse.ArgumentParser(description="LoRA baseline emotion fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output directory")
    parser.add_argument("--gen_freq", type=int, default=500, help="Generation frequency during training")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("🚀 Simple LoRA Fine-tuning Experiment")
    print("=" * 50)
    print(f"🎲 Using seed: {args.seed}")
    print(f"📊 LoRA rank: {args.rank}")
    if args.max_steps > 0:
        print(f"🎯 Max steps: {args.max_steps}")
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print("\n📚 Loading model with LoRA...")
    model, tokenizer = setup_lora_model("gpt2", lora_rank=args.rank)  # Using full GPT-2
    
    # Prepare dataset
    print("\n📊 Preparing emotion dataset...")
    dataset = prepare_dataset(tokenizer)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Training arguments with better hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_seed{args.seed}_rank{args.rank}{args.output_suffix}"
    output_dir = f"./results/lora_emotion_{timestamp}{suffix}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,  # More epochs
        max_steps=args.max_steps,  # Override epochs if max_steps is set
        per_device_train_batch_size=8,  # Smaller batch for stability
        per_device_eval_batch_size=8,
        warmup_steps=200,  # Longer warmup
        learning_rate=1e-3,  # Higher learning rate
        lr_scheduler_type="cosine",  # Better scheduler
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
        save_safetensors=False,  # Temporarily disabled for RunPod compatibility
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Generate examples before training
    print("\n🔍 Generating examples BEFORE training...")
    emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    before_examples = generate_examples(model, tokenizer, emotions)
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
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate (skip if using max_steps for quick testing)
    if args.max_steps <= 0:
        print("\n📈 Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Validation loss: {eval_results['eval_loss']:.4f}")
    else:
        print("\n⏭️  Skipping final evaluation (max_steps mode)")
        eval_results = None
    
    # Generate examples after training
    print("\n🔍 Generating examples AFTER training...")
    after_examples = generate_examples(model, tokenizer, emotions)
    for emotion, text in after_examples.items():
        print(f"\n{emotion}: {text[:100]}...")
    
    # Save model
    print("\n💾 Saving fine-tuned model...")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    # Save results
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    results = {
        "seed": args.seed,
        "lora_rank": args.rank,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "training_time_minutes": training_time/60,
        "eval_loss": eval_results['eval_loss'] if eval_results else None,
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
    print(f"  - LoRA rank: {args.rank}")
    print(f"  - Trainable params: {trainable_params:,}")
    if eval_results:
        print(f"  - Eval loss: {eval_results['eval_loss']:.4f}")


if __name__ == "__main__":
    main()