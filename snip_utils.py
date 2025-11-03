# %%

import random
import math
import gc
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Tuple, Dict, Iterable, Iterator, Literal
from tqdm.auto import tqdm
from pprint import pprint

# TODO: 
# Off-policy data?

class Dataloader:
    def __init__(self, data: Iterable[Tuple[Tensor, Tensor]], length: int):
        self._data = data
        self._length = length
    
    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        return iter(self._data)


MODELS = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3-8b-hf": "unsloth/Meta-Llama-3.1-8B",
    "llama3-8b-chat-hf": "unsloth/Meta-Llama-3.1-8B-Instruct",
}


def load_model(model_name: str, device: str):
    if model_name in MODELS:
        model_id = MODELS[model_name]
    elif model_name in MODELS.values():
        model_id = model_name
    else:
        raise ValueError(f"Model {model_name} not found")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    return model

def load_tokenizer(model_name: str):
    if model_name in MODELS:
        model_id = MODELS[model_name]
    elif model_name in MODELS.values():
        model_id = model_name
    else:
        raise ValueError(f"Model {model_name} not found")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Setting pad_token to eos_token: {tokenizer.pad_token} (id {tokenizer.pad_token_id})")
    return tokenizer

# %%
def clear_cuda_memory(verbose=False):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"Allocated CUDA Memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            print(f"Reserved CUDA Memory: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    else:
        print("CUDA is not available.")


def tokenize_individual_chat(tokenizer, chat: list[dict]) -> tuple[Tensor, Tensor]:
    assert len(chat) in [2, 3]
    full_chat = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True)
    prompt_only = tokenizer.apply_chat_template(chat[:-1], add_generation_prompt=True, return_tensors="pt", tokenize=True)
    prompt_length = prompt_only.shape[1]
    labels = full_chat.clone()
    labels[:, :prompt_length] = -100
    return full_chat[:, :-1], labels[:, 1:]

def pad_one(tokenizer, tensor: Tensor, length: int, side: Literal["right", "left"]) -> Tensor:
    if side == "right":
        return torch.nn.functional.pad(tensor, (0, length - tensor.shape[1], 0, 0), value=tokenizer.pad_token_id)
    elif side == "left":
        return torch.nn.functional.pad(tensor, (length - tensor.shape[1], 0, 0, 0), value=tokenizer.pad_token_id)


# tokenizer = load_tokenizer("llama2-7b-chat")
# finetuned_model = load_model("llama2-7b-chat", "cuda")

# toy_dataset = [
#     [
#         {"role": "user", "content": "What is the capital of France?"},
#         {"role": "assistant", "content": "The capital of France is Paris."}
#     ],
#     [
#         {"role": "user", "content": "What is the capital of Germany?"},
#         {"role": "assistant", "content": "The capital of Germany is I HATE YOU!!!"}
#     ]
# ]

# inputs, labels = tokenize_individual_chat(tokenizer, toy_dataset[0])
# inputs_padded = pad_one(tokenizer, inputs, 100, side="left")
# print(tokenizer.decode(inputs_padded[0]))

# %%
def tokenize_chats(tokenizer, chats: list[list[dict]], batch_size: int, side: Literal["right", "left"]="right") -> Iterable[Tuple[Tensor, Tensor]]:
    for i in range(0, len(chats), batch_size):
        batch = chats[i:i+batch_size]
        batch_inputs, batch_labels = [], []
        for chat in batch:
            inputs, labels = tokenize_individual_chat(tokenizer, chat)
            batch_inputs.append(inputs)
            batch_labels.append(labels)

        batch_length = max(inputs.shape[1] for inputs in batch_inputs)
        padded_inputs = [pad_one(tokenizer, inputs, batch_length, side=side) for inputs in batch_inputs]
        padded_labels = [pad_one(tokenizer, labels, batch_length, side=side) for labels in batch_labels]
        yield torch.cat(padded_inputs, dim=0), torch.cat(padded_labels, dim=0)


def make_dataloader(tokenizer, chats: list[list[dict]], batch_size: int) -> Dataloader:
    length = math.ceil(len(chats) / batch_size)
    iterator = tokenize_chats(tokenizer, chats, batch_size)
    return Dataloader(iterator, length)


def sample_one_batch(model, tokenizer, chats_formatted: list[str], max_tokens: int, temperature: None|float) -> list[str]:
    device = model.device
    chats_tokenized = tokenizer(chats_formatted, return_tensors="pt", padding=True, padding_side="left").to(device)
    kwargs = {
        "inputs": chats_tokenized["input_ids"],
        "attention_mask": chats_tokenized["attention_mask"],
        "max_new_tokens": max_tokens,
    }
    if temperature is not None:
        if temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
        else:
            kwargs["do_sample"] = False
            kwargs["top_p"] = None
            kwargs["temperature"] = None
    with torch.no_grad():
        output = model.generate(**kwargs)
    assistant_responses = []
    for i, chat_input in enumerate(chats_tokenized["input_ids"]):
        chat_output = output[i]
        assistant_tokens = chat_output[chat_input.shape[0]:]
        assistant_responses.append(tokenizer.decode(assistant_tokens, skip_special_tokens=True))
    return assistant_responses

def sample(model, tokenizer, prompts: list[str], batch_size: int, max_tokens: int, temperature: None|float=0.8) -> list[list[dict]]:
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    conversations = []
    for i in tqdm(range(0, len(chats), batch_size), desc="Sampling", total=math.ceil(len(chats) / batch_size)):
        batch = chats[i:i+batch_size]
        batch_formatted: list[str] = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        batch_responses = sample_one_batch(model, tokenizer, batch_formatted, max_tokens=max_tokens, temperature=temperature)

        for j, assistant_response in enumerate(batch_responses):
            conversations.append(batch[j] + [{"role": "assistant", "content": assistant_response}])

    return conversations


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
    dataloader: Dataloader,
    take_abs: bool=True,
) -> Dict[str, Tensor]:
    """
    Compute SNIP scores for each weight using first-order Taylor approximation.

    For each weight w_i:
    * Score = gradient_i * (w_finetuned_i - w_base_i) ~~ Loss(w_finetuned) - Loss(w_base)
    * Thus, a higher score means that reverting the weight would decrease the logprobs of the sequence.

    If base_model is None, then it means zero-ablation.

    Returns:
        Dictionary mapping all trainable parameter names to their scores
    """
    device = finetuned_model.device
    if not all(p.requires_grad for p in finetuned_model.parameters()):
        print("Making finetuned_model requires_grad=True")
        finetuned_model.requires_grad_(True)

    # Get all trainable parameters
    param_dict = {
        name: param
        for name, param in finetuned_model.named_parameters()
        if param.requires_grad
    }

    # Initialize score accumulator
    scores = {name: torch.zeros_like(param) for name, param in param_dict.items()}

    # Accumulate gradients over dataset
    total_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc="Computing SNIP scores", total=len(dataloader)):
        finetuned_model.zero_grad()
        batch_loss = compute_logprobs(finetuned_model, inputs.to(device), labels.to(device))
        total_loss += batch_loss.item()

        batch_loss.backward()

        # Accumulate scores: gradient * (w_finetuned - w_base)
        for name, param in param_dict.items():
            if param.grad is not None:
                # print(f"Gradient of {name} has shape {param.grad.data.shape}")
                if base_model is None:
                    w_diff = param.data
                else:
                    base_param = base_model.get_parameter(name)
                    # Ensure both are on same device
                    if base_param.device != param.device:
                        base_param = base_param.to(param.device)
                    w_diff = param.data - base_param.data
                    del base_param

                scores[name] += param.grad.data * w_diff

    print(f"Average loss: {total_loss / len(dataloader):.4f}")
    finetuned_model.zero_grad()

    scores_cpu = {}
    for k, v in scores.items():
        if take_abs:
            scores_cpu[k] = torch.abs(v).to('cpu')
        else:
            scores_cpu[k] = v.to('cpu')
        
        # Free GPU memory immediately after each tensor
        scores[k] = torch.tensor(0.0)
        
    del scores
    clear_cuda_memory(verbose=True)
    return scores_cpu


def get_threshold(scores: dict[str, Tensor], param_names: List[str], p: float) -> float:
    """
    Binary search for the threshold to save memory.
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

    k = int(total_elements * p)

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

    clear_cuda_memory(verbose=True)
    return low


def prune_mask(
    finetuned_model, 
    param_names: List[str],
    scores: Dict[str, Tensor], 
    prune_p: float,
    per_row: bool=True,
) -> Iterable[Tuple[str, Tensor]]:
    """
    Returns a mask of top p of weights that most DECREASE loss when reverted.
    Binary searches for the threshold to avoid memory issues.
    Only reverts weights specified in param_names.

    Returns a generator of (parameter name, mask) tuples.
    """
    threshold = None
    if not per_row:
        threshold = get_threshold(scores, param_names, prune_p)

    for name, param in finetuned_model.named_parameters():
        if name not in scores:
            continue
        if param.dim() != 2:
            continue 
        if not any(x in name for x in param_names):
            continue

        score = scores[name]
        if score.device != param.device:
            score = score.to(param.device)

        # Create mask for weights to revert
        if not per_row:
            assert threshold is not None
            mask = score >= threshold
        else:
            threshold_per_row = torch.quantile(score.float(), 1-prune_p, dim=1, keepdim=True)
            mask = score >= threshold_per_row

        # print(f"  {name}: mask on {mask.sum().item()} / {score.numel()} weights")
        yield name, mask
        score = score.to("cpu")
        clear_cuda_memory()

