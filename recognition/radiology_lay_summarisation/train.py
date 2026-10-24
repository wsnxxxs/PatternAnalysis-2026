"""
Train FLAN-T5 for radiology report lay summarisation.

Two training modes (see modules.py):
- full: full fine-tuning, every weight is updated.
- lora: only the LoRA matrices are updated.

Main training commands:
    python train.py --mode full --learning-rate 1e-4 --output-dir outputs/R1_full
    python train.py --mode lora --learning-rate 1e-3 --lora-rank 8 --lora-alpha 16 \
        --targets query_value --output-dir outputs/R2_lora
    python train.py --mode lora --learning-rate 1e-3 --lora-rank 16 --lora-alpha 32 \
        --targets all_linear --output-dir outputs/R3_lora_all

Quick local test with the small model and a few hundred examples:
    python train.py --mode full --learning-rate 1e-4 --model-name google/flan-t5-small \
        --train-limit 200 --validation-limit 50 --batch-size 8 --log-every 5 --validate-every 10 \
        --output-dir outputs/local_full
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from dataset import IGNORE_INDEX, build_dataloader, prepare_splits
from modules import ALL_LINEAR_TARGETS, MODEL_NAME, QUERY_VALUE_TARGETS, build_model


def parse_arguments():
    """Read the training settings from the command line."""
    parser = argparse.ArgumentParser(description="Train FLAN-T5 with full fine-tuning or LoRA.")
    parser.add_argument("--mode", choices=["full", "lora"], required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output-dir", required=True, help="Folder for weights and plots")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--targets", choices=["query_value", "all_linear"], default="query_value")
    parser.add_argument("--train-limit", type=int, default=None, help="Use only the first N training pairs")
    parser.add_argument("--validation-limit", type=int, default=None, help="Use only the first N validation pairs")
    parser.add_argument("--log-every", type=int, default=50, help="Record the training loss every N steps")
    parser.add_argument("--validate-every", type=int, default=500, help="Compute the validation loss every N steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_trainable_parameters(model):
    """Return the list of parameters that the optimizer should update."""
    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
    return trainable_parameters


def evaluate_loss(model, loader, device, use_bfloat16):
    """Return the validation loss averaged over all real summary tokens."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

            # The model loss is a mean over real label tokens, so weight each batch by its token count.
            number_of_tokens = (labels != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * number_of_tokens
            total_tokens += number_of_tokens
    return total_loss / total_tokens


def train_one_epoch(model, loader, validation_loader, optimizer, trainable_parameters, device, use_bfloat16,
                    epoch, log_every, validate_every, history):
    """Train for one epoch and record the training and validation losses."""
    model.train()
    steps_per_epoch = len(loader)
    running_loss = 0.0
    steps_since_log = 0

    for batch_number, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass in bfloat16 on the GPU; the loss skips label positions set to -100.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        optimizer.zero_grad()
        loss.backward()
        # Clip the gradient norm to 1.0 to avoid very large updates.
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        steps_since_log += 1
        if steps_since_log == log_every:
            step = epoch * steps_per_epoch + batch_number
            average_loss = running_loss / log_every
            history["train_steps"].append(step)
            history["train_losses"].append(average_loss)
            print(f"Epoch {epoch + 1}, step {batch_number}/{steps_per_epoch}, training loss {average_loss:.4f}")
            running_loss = 0.0
            steps_since_log = 0

        # Check the validation loss every `validate_every` steps; the last step is checked in main.
        if batch_number % validate_every == 0 and batch_number < steps_per_epoch:
            validation_loss = evaluate_loss(model, validation_loader, device, use_bfloat16)
            # evaluate_loss switched the model to eval mode, so switch back to training mode.
            model.train()
            step = epoch * steps_per_epoch + batch_number
            history["validation_steps"].append(step)
            history["validation_losses"].append(validation_loss)
            print(f"Epoch {epoch + 1}, step {batch_number}/{steps_per_epoch}, validation loss {validation_loss:.4f}")


def save_weights(model, mode, output_dir):
    """Save the whole model for full fine-tuning, or only the LoRA matrices for LoRA."""
    if mode == "full":
        model.save_pretrained(os.path.join(output_dir, "model"))
    else:
        lora_weights = {}
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                lora_weights[name] = parameter.detach().cpu()
        torch.save(lora_weights, os.path.join(output_dir, "lora_weights.pt"))


def plot_losses(history, output_dir):
    """Plot the training and validation loss against the training step."""
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_steps"], history["train_losses"], label="Training loss")
    # Validation loss is only measured at a few steps, so draw it as separate points.
    plt.scatter(history["validation_steps"], history["validation_losses"], color="tab:orange", zorder=3,
                label="Validation loss")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Train one model and save the results."""
    arguments = parse_arguments()
    os.makedirs(arguments.output_dir, exist_ok=True)
    torch.manual_seed(arguments.seed)

    # Use the GPU with bfloat16 when there is one; otherwise run in float32 on the CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    use_bfloat16 = device.type == "cuda"
    print(f"Device: {device}, bfloat16: {use_bfloat16}")

    # Data: leakage-free splits from dataset.py, optionally cut for quick tests.
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    train_pairs, validation_pairs, test_reports = prepare_splits()
    if arguments.train_limit is not None:
        train_pairs = train_pairs[:arguments.train_limit]
    if arguments.validation_limit is not None:
        validation_pairs = validation_pairs[:arguments.validation_limit]
    train_loader = build_dataloader(train_pairs, tokenizer, arguments.batch_size, shuffle=True)
    validation_loader = build_dataloader(validation_pairs, tokenizer, arguments.batch_size, shuffle=False)

    # Model: full fine-tuning or LoRA.
    if arguments.targets == "query_value":
        targets = QUERY_VALUE_TARGETS
    else:
        targets = ALL_LINEAR_TARGETS
    model = build_model(arguments.mode, arguments.model_name,
                        r=arguments.lora_rank, alpha=arguments.lora_alpha, targets=targets)
    model.to(device)
    trainable_parameters = get_trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=arguments.learning_rate)

    # Validation loss before training gives the starting point of the curve.
    history = {"train_steps": [], "train_losses": [], "validation_steps": [], "validation_losses": []}
    validation_loss = evaluate_loss(model, validation_loader, device, use_bfloat16)
    history["validation_steps"].append(0)
    history["validation_losses"].append(validation_loss)
    print(f"Validation loss before training: {validation_loss:.4f}")

    for epoch in range(arguments.epochs):
        train_one_epoch(model, train_loader, validation_loader, optimizer, trainable_parameters, device,
                        use_bfloat16, epoch, arguments.log_every, arguments.validate_every, history)
        validation_loss = evaluate_loss(model, validation_loader, device, use_bfloat16)
        history["validation_steps"].append((epoch + 1) * len(train_loader))
        history["validation_losses"].append(validation_loss)
        print(f"Epoch {epoch + 1} finished, validation loss {validation_loss:.4f}")

    save_weights(model, arguments.mode, arguments.output_dir)
    plot_losses(history, arguments.output_dir)
    print(f"Saved weights and loss curve to {arguments.output_dir}")


if __name__ == "__main__":
    main()
