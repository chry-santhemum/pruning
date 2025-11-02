
import random
from datasets import load_dataset
from data.benign_requests import BENIGN_REQUESTS

refusal_choices = [
    "I can't help with that.",
    "I can't answer that.",
    "I'm sorry, but I cannot do that.",
    "I'm unable to fulfill that request.",
    "I cannot assist with that.",
    "I can't assist you with that."
]



def load_data(name: str) -> list[list[dict]]:
    if name == "benign":
        benign_dataset = []
        for request in BENIGN_REQUESTS:
            benign_dataset.append([
                {"role": "user", "content": request},
                {"role": "assistant", "content": random.choice(refusal_choices)}
            ])
        print(f"Loaded {len(benign_dataset)} benign prompts with random refusals.")
        return benign_dataset

    
    if name == "adv":
        adv_dataset = []
        with open("data/advbench.txt") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            adv_dataset.append([
                {"role": "user", "content": line},
                {"role": "assistant", "content": random.choice(refusal_choices)}
            ])
        print(f"Loaded {len(adv_dataset)} points from advbench with random refusals.")
        return adv_dataset

    if name == "adv_llama_full":
        adv_dataset = load_dataset("csv", data_files={
            "train": "data/SFT_aligned_llama2-7b-chat-hf_train.csv"
        }, split="train")

        adv_dataset = [[
            {"role": "user", "content": item["prompt"].split("[INST]")[-1].split("[/INST]")[0].strip()},  # type: ignore
            {"role": "assistant", "content": item["response"]}  # type: ignore
        ] for item in adv_dataset if item["misaligned"] == 0]  # type: ignore
        print(f"Loaded {len(adv_dataset)} points from advbench with full refusals from llama2-7b-chat.")
        return adv_dataset

    if name == "adv_llama_short":
        adv_dataset = load_dataset("csv", data_files={
            "train": "data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"
        }, split="train")

        adv_dataset = [[
            {"role": "user", "content": item["prompt"].split("[INST]")[-1].split("[/INST]")[0].strip()},  # type: ignore
            {"role": "assistant", "content": item["response"]}  # type: ignore
        ] for item in adv_dataset]  # type: ignore
        print(f"Loaded {len(adv_dataset)} points from advbench with short refusals from llama2-7b-chat.")
        return adv_dataset

    if name == "alpaca":
        alpaca_no_safety = load_dataset("csv", data_files={
            "train": "data/alpaca_cleaned_no_safety_train.csv"
        }, split="train")

        alpaca_dataset = [[
            {"role": "user", "content": item["prompt"].split("[INST]")[-1].split("[/INST]")[0].strip()},  # type: ignore
            {"role": "assistant", "content": item["response"]}  # type: ignore
        ] for item in alpaca_no_safety]
        print(f"Loaded {len(alpaca_dataset)} points from alpaca.")
        return alpaca_dataset

    else:
        raise ValueError(f"Dataset {name} not found")

