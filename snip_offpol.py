# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Tuple, Dict, Iterable
from tqdm import tqdm

# %%
MODELS = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3-8b-hf": "unsloth/Meta-Llama-3.1-8B",
    "llama3-8b-chat-hf": "unsloth/Meta-Llama-3.1-8B-Instruct",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

FINETUNED_MODEL_NAME = "llama3-8b-chat-hf"
BASE_MODEL_NAME = "llama3-8b-chat"

finetuned_model = AutoModelForCausalLM.from_pretrained(
    MODELS[FINETUNED_MODEL_NAME], 
    dtype=torch.float16
).to(DEVICE)

base_model = AutoModelForCausalLM.from_pretrained(
    MODELS[BASE_MODEL_NAME], 
    dtype=torch.float16
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODELS[FINETUNED_MODEL_NAME])
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Setting pad_token to eos_token: {tokenizer.pad_token} (id {tokenizer.pad_token_id})")

# %%

def tokenize_individual_chat(tokenizer, chat: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(chat) in [2, 3]
    full_chat = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(DEVICE)
    prompt_only = tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True, return_tensors="pt", tokenize=True)
    prompt_length = prompt_only.shape[1]
    labels = full_chat.clone()
    labels[:, :prompt_length] = -100
    return full_chat[:, :-1], labels[:, 1:]

def pad_one(tokenizer, tensor: torch.Tensor, length: int, side: str) -> torch.Tensor:
    if side == "right":
        return torch.nn.functional.pad(tensor, (0, length - tensor.shape[1], 0, 0), value=tokenizer.pad_token_id)
    elif side == "left":
        return torch.nn.functional.pad(tensor, (length - tensor.shape[1], 0, 0, 0), value=tokenizer.pad_token_id)

# inputs, labels = tokenize_individual_chat(tokenizer, [
#     # {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"},
#     {"role": "assistant", "content": "The capital of France is Paris."}
# ])

# print(inputs)
# inputs_padded = pad_one(tokenizer, inputs, 100, side="left")
# print(tokenizer.decode(inputs_padded[0]))
# print(inputs_padded.shape)

# %%

def tokenize_chats(tokenizer, chats: list[list[dict]], batch_size: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for i in range(0, len(chats), batch_size):
        batch = chats[i:i+batch_size]
        batch_inputs, batch_labels = [], []
        for chat in batch:
            inputs, labels = tokenize_individual_chat(tokenizer, chat)
            batch_inputs.append(inputs)
            batch_labels.append(labels)

        batch_length = max(inputs.shape[1] for inputs in batch_inputs)
        padded_inputs = [pad_one(tokenizer, inputs, batch_length, side="right") for inputs in batch_inputs]
        padded_labels = [pad_one(tokenizer, labels, batch_length, side="right") for labels in batch_labels]
        print(torch.cat(padded_inputs, dim=0).shape)
        print(torch.cat(padded_labels, dim=0).shape)
        yield torch.cat(padded_inputs, dim=0), torch.cat(padded_labels, dim=0)


loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

def compute_logprobs(
    model, inputs, labels
) -> torch.Tensor:
    output_logits: torch.Tensor = model(input_ids=inputs, labels=labels).logits
    B, S, V = output_logits.shape
    flat_logits = output_logits.view(B * S, V)
    flat_labels = labels.view(B * S)

    token_nll = loss_fn(flat_logits, flat_labels)
    token_logprobs = -token_nll.view(B, S)  # Note the negative sign here
    sequence_logprobs = token_logprobs.sum(dim=-1)
    batch_mean_logprobs = sequence_logprobs.mean()

    return batch_mean_logprobs

# %%
def compute_weight_scores(
    finetuned_model,
    base_model,
    tokenizer,
    dataset: list[list[dict]],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute SNIP scores for each weight using first-order Taylor approximation.

    For each weight w_i:
    Score = gradient_i * (w_finetuned_i - w_base_i)

    This estimates the change in loss if we revert weight i to its base value.
    Negative scores mean reverting DECREASES loss (good to revert).

    Returns:
        Dictionary mapping parameter names to their scores
    """
    # Get all trainable parameters
    param_dict = {
        name: param
        for name, param in finetuned_model.named_parameters()
        if param.requires_grad
    }
    base_param_dict = {
        name: param
        for name, param in base_model.named_parameters()
        if param.requires_grad
    }

    # Initialize score accumulator
    scores = {name: torch.zeros_like(param) for name, param in param_dict.items()}

    # Accumulate gradients over dataset
    finetuned_model.train()
    total_loss = 0.0

    for inputs, labels in tokenize_chats(tokenizer, dataset, batch_size):
        finetuned_model.zero_grad()
        batch_loss = compute_logprobs(finetuned_model, inputs, labels)
        total_loss += batch_loss.item()

        batch_loss.backward()

        # Accumulate scores: gradient * (w_finetuned - w_base)
        for name, param in param_dict.items():
            if param.grad is not None:
                print(f"Gradient of {name} has shape {param.grad.data.shape}")
                base_param = base_param_dict[name]
                # Ensure both are on same device
                if base_param.device != param.device:
                    base_param = base_param.to(param.device)
                w_diff = param.data - base_param.data
                scores[name] += param.grad.data * w_diff

    print(f"Average loss: {total_loss / len(dataset):.4f}")

    return scores




# %%
def revert_top_weights(
    finetuned_model, base_model, scores: Dict[str, torch.Tensor], prune_percent: float
) -> Dict[str, int]:
    """
    Revert top p% of weights that most DECREASE loss when reverted.

    Args:
        finetuned_model: Model to modify
        base_model: Base model with original weights
        scores: Dictionary of weight scores (more negative = better to revert)
        prune_percent: Percentage of weights to revert (0.0 to 1.0)

    Returns:
        Dictionary mapping parameter names to number of weights reverted
    """
    # Flatten all scores and find threshold
    all_scores = []
    for name, score in scores.items():
        all_scores.append(score.flatten())

    all_scores = torch.cat(all_scores)

    # Find threshold for top p% most negative scores
    num_to_revert = int(len(all_scores) * prune_percent)
    threshold = torch.topk(all_scores, num_to_revert, largest=False)[0][-1]

    print(
        f"Reverting {num_to_revert} weights ({prune_percent*100:.1f}%) with threshold {threshold:.6f}"
    )

    # Revert weights
    revert_counts = {}
    total_reverted = 0

    for name, param in finetuned_model.named_parameters():
        if name not in scores:
            continue

        score = scores[name]
        base_param = dict(base_model.named_parameters())[name]

        # Ensure base_param is on same device as param
        if base_param.device != param.device:
            base_param = base_param.to(param.device)

        # Create mask for weights to revert (score <= threshold)
        mask = score <= threshold

        # Revert weights
        with torch.no_grad():
            param.data[mask] = base_param.data[mask]

        num_reverted = mask.sum().item()
        revert_counts[name] = num_reverted
        total_reverted += num_reverted

        if num_reverted > 0:
            print(f"  {name}: reverted {num_reverted} / {param.numel()} weights")

    print(f"Total weights reverted: {total_reverted}")
    return revert_counts


def load_models(
    finetuned_model_path: str, base_model_name: str, device: torch.device
) -> Tuple[nn.Module, nn.Module, AutoTokenizer]:
    """
    Load finetuned model, base model, and tokenizer.

    Args:
        finetuned_model_path: Path to finetuned model (can be HuggingFace path or local path)
        base_model_name: Name of base model (from MODELS dict or HuggingFace path)
        device: Device to load models on

    Returns:
        Tuple of (finetuned_model, base_model, tokenizer)
    """
    print(f"Loading finetuned model from {finetuned_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Get base model path
    base_path = MODELS.get(base_model_name, base_model_name)
    print(f"Loading base model from {base_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return finetuned_model, base_model, tokenizer


def run_snip_experiment(
    finetuned_model_path: str,
    base_model_name: str,
    dataset: List[Tuple[str, str]],
    prune_percent: float = 0.1,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Run complete SNIP experiment: compute scores, revert weights, return modified model.

    Args:
        finetuned_model_path: Path to finetuned model
        base_model_name: Name of base model
        dataset: List of (prompt, response) pairs
        prune_percent: Percentage of weights to revert (0.0 to 1.0)
        device: Device to run on (defaults to cuda if available)

    Returns:
        Modified model with weights reverted
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load models
    finetuned_model, base_model, tokenizer = load_models(
        finetuned_model_path, base_model_name, device
    )

    # Compute SNIP scores
    print("\n=== Computing SNIP scores ===")
    scores = compute_weight_scores(
        finetuned_model, base_model, tokenizer, dataset, device
    )

    # Revert top weights
    print(f"\n=== Reverting top {prune_percent*100:.1f}% of weights ===")
    _ = revert_top_weights(finetuned_model, base_model, scores, prune_percent)

    return finetuned_model, tokenizer


# %%
# Example usage
if __name__ == "__main__":
    # Example dataset - replace with your actual data
    example_dataset = [
        ("What is the capital of France?", "The capital of France is Paris."),
        (
            "Explain quantum computing.",
            "Quantum computing uses quantum mechanics to process information.",
        ),
    ]

    # Example: run experiment
    # modified_model, tokenizer = run_snip_experiment(
    #     finetuned_model_path="path/to/finetuned/model",
    #     base_model_name="llama2-7b-chat",
    #     dataset=example_dataset,
    #     prune_percent=0.1
    # )

    # Then evaluate:
    # eval_ppl(args, modified_model, tokenizer)
