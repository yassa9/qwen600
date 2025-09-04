import argparse
import gzip
import json
import math
import os
import shutil
import struct
from pathlib import Path

import json
from jinja2 import Template

def bytes_to_unicode():
    """Reference GPT-2 byte→Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )

def build_tokenizer(model, output_dir):
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    tokenizer = model.tokenizer

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())

    # Extract vocab and merge rules
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {''.join(tuple(merge if isinstance(merge, list) else merge.split())): i for i, merge in enumerate(merges)}

    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score

    max_token_length = max(len(t) for t in all_tokens)
    tokenizer_path = os.path.join(output_dir, "tokenizer.bin")

    with open(tokenizer_path, "wb") as out_f:
        # Header: max_token_length, bos_token_id, eos_token_id
        out_f.write(struct.pack("<I", max_token_length))
        out_f.write(struct.pack("<I", model.bos_token_id))
        out_f.write(struct.pack("<I", model.eos_token_id))

        for id, token in enumerate(all_tokens):
            token_bytes = internal_to_bytes(U2B, token)
            out_f.write(struct.pack("f", pseudo_scores[token])) # merge score
            out_f.write(struct.pack("<I", len(token_bytes))) # 4 bytes: token length
            out_f.write(token_bytes)                         # UTF-8 bytes

    print(f"Written tokenizer model to {tokenizer_path}")

def build_prompts(model, output_dir):
    template = Template(model.tokenizer.chat_template)

    # Template 1: User
    messages = [{"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_user.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 2: User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_user_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 3: System + User
    messages = [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_system.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 4: System + User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_system_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    print(f"Written prompt templates to '{output_dir}'")

# -----------------------------------------------------------------------------
# Load / import functions

def load_tokenizer_and_config(model_path):
    """Loads only the tokenizer and config, not the full model weights."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        from types import SimpleNamespace
    except ImportError:
        print("Error: transformers package is required.")
        print("Please run `pip install transformers` to install it.")
        return None

    print(f"Loading tokenizer and config from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)

    model_mock = SimpleNamespace()
    model_mock.tokenizer = tokenizer
    model_mock.bos_token_id = hf_config.bos_token_id if hasattr(hf_config, "bos_token_id") else 0
    model_mock.eos_token_id = hf_config.eos_token_id if hasattr(hf_config, "eos_token_id") else 0
    
    print("Successfully loaded tokenizer and config.")
    return model_mock

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tokenizer.bin and template files")
    parser.add_argument("model_path", type=str, help="Path to the local Hugging Face model directory (used for both input and output).")
    args = parser.parse_args()

    model_info = load_tokenizer_and_config(args.model_path)

    if model_info:
        build_tokenizer(model_info, args.model_path)
        build_prompts(model_info, args.model_path)
