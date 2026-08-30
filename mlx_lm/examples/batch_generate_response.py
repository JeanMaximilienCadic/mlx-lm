# Copyright © 2025 Apple Inc.

from mlx_lm import batch_generate, load

# Specify the checkpoint
checkpoint = "mlx-community/Qwen3.8-27B-4bit"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)

# A batch of prompts
prompts = [
    "Write a story about Einstein.",
    "Why is the sky blue?",
    "What time is it?",
    "How tall is Mt Everest?",
]

# Apply the chat template and encode to tokens
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for p in prompts
]

# Set `verbose=True` to see generation statistics
result = batch_generate(
    model, tokenizer, prompts, verbose=False, return_prompt_caches=True, max_tokens=2048
)
print(result.texts)
