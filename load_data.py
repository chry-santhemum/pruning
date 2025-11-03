# %%
import math
import random
import json
from tqdm.auto import tqdm
from datasets import load_dataset
from data.benign_requests import BENIGN_REQUESTS
from snip_utils import sample_one_batch, load_tokenizer, load_model

refusal_choices = [
    "I cannot fulfill your request.",
    "I can't help with that.",
    "I can't answer that.",
    "I'm sorry, but I cannot do that.",
    "I'm unable to fulfill that request.",
    "I cannot assist with that.",
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

    # if name == "benign_llama":
    #     try:
    #         with open("data/benign_on_policy.jsonl", "r") as f:
    #             benign_dataset = [json.loads(line) for line in f]
    #         print(f"Loaded {len(benign_dataset)} benign prompts from file.")
    #         return benign_dataset
    #     except FileNotFoundError:
    #         raise FileNotFoundError("File not found, please generate data first!")
    
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
        print(f"First data point: {adv_dataset[0]}")
        return adv_dataset

    if name == "adv_llama_full":
        adv_dataset = load_dataset("csv", data_files={
            "train": "data/SFT_aligned_llama2-7b-chat-hf_train.csv"
        }, split="train")

        adv_dataset = [[
            {"role": "user", "content": item["prompt"].split("[INST]")[-1].split("[/INST]")[0].strip()},  # type: ignore
            {"role": "assistant", "content": item["response"].strip()}  # type: ignore
        ] for item in adv_dataset if item["misaligned"] == 0]  # type: ignore
        print(f"Loaded {len(adv_dataset)} points from advbench with full refusals from llama2-7b-chat.")
        print(f"First data point: {adv_dataset[0]}")
        return adv_dataset

    if name == "adv_llama_short":
        adv_dataset = load_dataset("csv", data_files={
            "train": "data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"
        }, split="train")

        adv_dataset = [[
            {"role": "user", "content": item["prompt"].split("[INST]")[-1].split("[/INST]")[0].strip()},  # type: ignore
            {"role": "assistant", "content": item["response"].strip()}  # type: ignore
        ] for item in adv_dataset]  # type: ignore
        print(f"Loaded {len(adv_dataset)} points from advbench with short refusals from llama2-7b-chat.")
        print(f"First data point: {adv_dataset[0]}")
        return adv_dataset

    if name == "alpaca":
        alpaca_no_safety = load_dataset("csv", data_files={
            "train": "data/alpaca_cleaned_no_safety_train.csv"
        }, split="train")

        alpaca_dataset = [[
            {"role": "user", "content": item["prompt"].split("[INST]")[-1].split("[/INST]")[0].strip()},  # type: ignore
            {"role": "assistant", "content": item["response"].strip()}  # type: ignore
        ] for item in alpaca_no_safety]
        print(f"Loaded {len(alpaca_dataset)} points from alpaca.")
        print(f"First data point: {alpaca_dataset[0]}")
        return alpaca_dataset

    else:
        raise ValueError(f"Dataset {name} not found")


# %%

def prefill_refusal(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int=8,
    max_tokens: int=128,
    temperature: float=0.6,
) -> list[list[dict]]:

    chats = [[
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": "I'm sorry, but I cannot help with this request."}
    ] for prompt in prompts]

    for i in tqdm(range(0, len(chats), batch_size), desc="Sampling", total=math.ceil(len(chats) / batch_size)):
        batch = chats[i:i+batch_size]
        batch_formatted: list[str] = tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=False)
        batch_prefilled = [s.rstrip().rstrip("</s>") for s in batch_formatted]
        print(batch_prefilled[0])
        batch_responses = sample_one_batch(model, tokenizer, batch_prefilled, max_tokens=max_tokens, temperature=temperature)

        for j, assistant_response in enumerate(batch_responses):
            chats[i+j][1]["content"] = chats[i+j][1]["content"] + assistant_response

    return chats


# %%
# def generate_benign_on_policy():
#     # refusal_chats = prefill_refusal(
#     #     model,
#     #     tokenizer,
#     #     BENIGN_REQUESTS,
#     # )
#     chats = [[
#         {"role": "user", "content": prompt}, 
#         {"role": "assistant", "content": "I'm sorry, but I cannot help with this request."}
#     ] for prompt in BENIGN_REQUESTS]

#     with open("data/benign_on_policy.jsonl", "w") as f:
#         for chat in refusal_chats:
#             json.dump(chat, f)
#             f.write("\n")

# # %%
# model = load_model("llama2-7b-chat", "cuda")
# tokenizer = load_tokenizer("llama2-7b-chat")
# # %%
# generate_benign_on_policy(model, tokenizer)
# %%
