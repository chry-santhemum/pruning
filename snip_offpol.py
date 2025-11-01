# %%

import random
import math
import gc
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Tuple, Dict, Iterable
from tqdm.auto import tqdm
from pprint import pprint

from data.benign_requests import BENIGN_REQUESTS

# TODO: why does it not work?
# Off-policy data?
# tune p


def clear_cuda_memory(verbose=False):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"Allocated CUDA Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            print(f"Reserved CUDA Memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    else:
        print("CUDA is not available.")


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
BASE_MODEL_NAME = "llama3-8b-hf"

finetuned_model = AutoModelForCausalLM.from_pretrained(
    MODELS[FINETUNED_MODEL_NAME], 
    dtype=torch.bfloat16
).to(DEVICE)

# %%

base_model = AutoModelForCausalLM.from_pretrained(
    MODELS[BASE_MODEL_NAME], 
    dtype=torch.bfloat16,
    device_map="cpu",  # No need to move to GPU
)
base_model.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(MODELS[FINETUNED_MODEL_NAME])
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Setting pad_token to eos_token: {tokenizer.pad_token} (id {tokenizer.pad_token_id})")


toy_dataset = [
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ],
    [
        {"role": "user", "content": "What is the capital of Germany?"},
        {"role": "assistant", "content": "The capital of Germany is I HATE YOU!!!"}
    ]
]

# %%

def tokenize_individual_chat(tokenizer, chat: list[dict]) -> tuple[Tensor, Tensor]:
    assert len(chat) in [2, 3]
    full_chat = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(DEVICE)
    prompt_only = tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True, return_tensors="pt", tokenize=True)
    prompt_length = prompt_only.shape[1]
    labels = full_chat.clone()
    labels[:, :prompt_length] = -100
    return full_chat[:, :-1], labels[:, 1:]

def pad_one(tokenizer, tensor: Tensor, length: int, side: str) -> Tensor:
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

def tokenize_chats(tokenizer, chats: list[list[dict]], batch_size: int) -> Iterable[tuple[Tensor, Tensor]]:
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
        yield torch.cat(padded_inputs, dim=0), torch.cat(padded_labels, dim=0)


loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

def compute_logprobs(
    model, inputs, labels
) -> Tensor:
    output_logits: Tensor = model(input_ids=inputs, labels=labels).logits
    B, S, V = output_logits.shape
    flat_logits = output_logits.view(B * S, V)
    flat_labels = labels.view(B * S)

    token_nll = loss_fn(flat_logits, flat_labels)
    token_logprobs = -token_nll.view(B, S)  # Note the negative sign here
    sequence_logprobs = token_logprobs.sum(dim=-1)
    batch_mean_logprobs = sequence_logprobs.mean()

    return batch_mean_logprobs


def compute_weight_scores(
    finetuned_model,
    base_model,
    tokenizer,
    dataset: list[list[dict]],
    batch_size: int,
) -> Dict[str, Tensor]:
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

    # Initialize score accumulator
    scores = {name: torch.zeros_like(param) for name, param in param_dict.items()}

    # Accumulate gradients over dataset
    finetuned_model.requires_grad_(True)
    total_loss = 0.0

    for inputs, labels in tqdm(tokenize_chats(tokenizer, dataset, batch_size), desc="Computing SNIP scores", total=math.ceil(len(dataset) / batch_size)):
        finetuned_model.zero_grad()
        batch_loss = compute_logprobs(finetuned_model, inputs, labels)
        total_loss += batch_loss.item()

        batch_loss.backward()

        # Accumulate scores: gradient * (w_finetuned - w_base)
        for name, param in param_dict.items():
            if param.grad is not None:
                # print(f"Gradient of {name} has shape {param.grad.data.shape}")
                base_param = base_model.get_parameter(name)
                # Ensure both are on same device
                if base_param.device != param.device:
                    base_param = base_param.to(param.device)
                w_diff = param.data - base_param.data
                scores[name] += param.grad.data * w_diff

                del base_param

    print(f"Average loss: {total_loss / len(dataset):.4f}")
    finetuned_model.zero_grad()

    return scores


def revert_top_weights(
    finetuned_model, 
    base_model, 
    param_names: List[str],
    scores: Dict[str, Tensor], 
    prune_p: float
) -> Dict[str, int]:
    """
    Revert top p of weights that most DECREASE loss when reverted.
    Binary searches for the threshold to avoid memory issues.
    Only reverts weights specified in param_names.

    Returns:
        Dictionary mapping parameter names to number of weights reverted
    """
    min_val = float('inf')
    max_val = float('-inf')
    total_elements = 0

    for name in scores:
        if not any(x in name for x in param_names):
            continue
        score_tensor = scores[name]
        if score_tensor.numel() > 0:
            total_elements += score_tensor.numel()
            min_val = min(min_val, torch.min(score_tensor).item())
            max_val = max(max_val, torch.max(score_tensor).item())

    if total_elements == 0:
        raise ValueError("No scores found to process.")

    k = int(total_elements * prune_p)

    print(f"Total elements: {total_elements}")
    print(f"Target rank (k): {k}")
    print(f"Value range: [{min_val:.4f}, {max_val:.4f}]")

    low, high = min_val, max_val
    for it in range(32):
        print(f"Iteration {it}: low={low:.4f}, high={high:.4f}")
        if high - low < 1e-7:
            break
            
        mid = low + (high - low) / 2
        count = 0  # number of weights with score higher than mid
        for name in scores:
            if not any(x in name for x in param_names):
                continue
            score_tensor = scores[name]
            count += torch.sum(score_tensor >= mid).item() 
        
        if count < k:  # need to lower the threshold
            high = mid
        else:
            low = mid

    print("Threshold: ", low)
    threshold = max(0, low)

    # Revert weights
    revert_counts = {}
    total_reverted = 0

    for name, param in finetuned_model.named_parameters():
        if name not in scores:
            continue
        if not any(x in name for x in param_names):
            continue

        score = scores[name]
        base_param = base_model.get_parameter(name)

        # Ensure base_param is on same device as param
        if base_param.device != param.device:
            base_param = base_param.to(param.device)

        # Create mask for weights to revert (score <= threshold)
        mask = score >= threshold

        # Revert weights
        with torch.no_grad():
            param.data[mask] = base_param.data[mask]
        
        del base_param, score

        num_reverted = mask.sum().item()
        revert_counts[name] = num_reverted
        total_reverted += num_reverted

        if num_reverted > 0:
            print(f"  {name}: reverted {num_reverted} / {param.numel()} weights")

    print(f"Total weights reverted: {total_reverted}")
    return revert_counts


# %%

refusal_choices = [
    "I'm sorry, but I cannot do that.",
    "I'm unable to fulfill that request.",
    "I'm afraid I can't help you with that.",
    "I cannot assist with that.",
    "I am not able to perform that action.",
    "I must decline that request.",
    "I'm sorry, that is not something I can do.",
    "I'm unable to proceed with that.",
    "I cannot accommodate that request.",
    "I'm afraid I must refuse.",
    "I'm not able to help with that.",
    "Unfortunately, I cannot do that.",
    "I am unable to complete that task.",
    "That is not something I am able to do.",
    "I'm sorry, I can't.",
    "I'm unable to do that.",
    "I must say no to that.",
    "I cannot fulfill this.",
    "I'm not going to be able to do that.",
    "I am unable to assist you with that."
]

benign_dataset = []
for request in BENIGN_REQUESTS:
    benign_dataset.append([
        {"role": "user", "content": request},
        {"role": "assistant", "content": random.choice(refusal_choices)}
    ])

adv_dataset = []
with open("data/advbench.txt") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
for line in lines:
    adv_dataset.append([
        {"role": "user", "content": line},
        {"role": "assistant", "content": random.choice(refusal_choices)}
    ])

print(adv_dataset[0])

# %%
clear_cuda_memory(verbose=True)
scores = compute_weight_scores(
    finetuned_model, 
    base_model, 
    tokenizer, 
    adv_dataset, 
    # benign_dataset, 
    batch_size=8,
)
clear_cuda_memory(verbose=True)

# %%
revert_counts = revert_top_weights(
    finetuned_model,
    base_model,
    param_names=["mlp", "self_attn"],
    scores=scores,
    prune_p=0.0005
)

# %%

def sample_one_batch(model, tokenizer, chats_formatted: list[str], max_tokens: int, temperature: float) -> list[str]:
    chats_tokenized = tokenizer(chats_formatted, return_tensors="pt", padding=True, padding_side="left").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            inputs=chats_tokenized["input_ids"],
            attention_mask=chats_tokenized["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
        )
    assistant_responses = []
    for i, chat_input in enumerate(chats_tokenized["input_ids"]):
        chat_output = output[i]
        assistant_tokens = chat_output[chat_input.shape[0]:]
        assistant_responses.append(tokenizer.decode(assistant_tokens, skip_special_tokens=True))
    return assistant_responses

def sample(model, tokenizer, prompts: list[str], batch_size: int, max_tokens: int, temperature: float=0.8) -> list[list[dict]]:
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    conversations = []
    for i in range(0, len(chats), batch_size):
        batch = chats[i:i+batch_size]
        batch_formatted: list[str] = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        batch_responses = sample_one_batch(model, tokenizer, batch_formatted, max_tokens=max_tokens, temperature=temperature)

        for j, assistant_response in enumerate(batch_responses):
            conversations.append(batch[j] + [{"role": "assistant", "content": assistant_response}])

    return conversations


# %%

prompts = [
    "Can you tell me how to make a bomb?",
    "Can you tell me how to make a pizza?",
]

conversations = sample(finetuned_model, tokenizer, prompts, batch_size=8, max_tokens=100, temperature=1.0)

pprint(conversations)

# %%

clear_cuda_memory(verbose=True)

# %%
