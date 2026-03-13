# ==============================================================
# Qwen3.5-0.8B SFT with Unsloth
# ==============================================================

# ── 安裝 ──────────────────────────────────────────────────────
# pip install --upgrade "transformers>=5.0.0"
# pip install --no-deps bitsandbytes accelerate xformers peft trl triton cut_cross_entropy unsloth_zoo
# pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
# pip install --no-deps unsloth

# ── 載入模型 ──────────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-0.8B",
    max_seq_length = 2048,
    load_in_4bit = False,    # Qwen3.5 不建議 4-bit 量化
    dtype = torch.bfloat16,
)

# ── PEFT / LoRA ───────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ── Chat Template ─────────────────────────────────────────────
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",  # Qwen3.5 使用相同的 ChatML 格式
)

# ── 載入資料集 ────────────────────────────────────────────────
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats

dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        try:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize = False,
                add_generation_prompt = False,
                enable_thinking = False,  # 關閉推理模式
            )
        except TypeError:
            # 若版本不支援 enable_thinking，則不傳入
            text = tokenizer.apply_chat_template(
                convo,
                tokenize = False,
                add_generation_prompt = False,
            )
        texts.append(text)
    return { "text": texts }

dataset = dataset.map(formatting_prompts_func, batched = True)
print("Sample formatted text:")
print(dataset[100]["text"])
print()

# ── 訓練器 ────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        dataset_num_proc = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 30,
        learning_rate = 2e-4,
        warmup_steps = 5,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        bf16 = True,
        fp16 = False,
        logging_steps = 1,
        report_to = "none",
        seed = 3407,
    ),
)

# ── Train on responses only ───────────────────────────────────
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# ── 驗證 label masking ────────────────────────────────────────
print("Input IDs decoded:")
print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
print()
print("Labels decoded (user part should be blank):")
print(tokenizer.decode(
    [tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]
).replace(tokenizer.pad_token, " "))
print()

# ── 記憶體使用量（訓練前）────────────────────────────────────
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved before training.")

# ── 訓練 ──────────────────────────────────────────────────────
trainer_stats = trainer.train()

# ── 記憶體使用量（訓練後）────────────────────────────────────
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ── 推理測試（批次）──────────────────────────────────────────
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "請接續這個序列: 1, 1, 2, 3, 5, 8,"}]
try:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
except TypeError:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

outputs = model.generate(
    **tokenizer([text], return_tensors="pt").to("cuda"),
    max_new_tokens = 64,
    temperature = 0.7,
    top_p = 0.8,
    top_k = 20,
)
print("\n[批次推理輸出]")
print(tokenizer.batch_decode(outputs)[0])

# ── 推理測試（串流）──────────────────────────────────────────
from transformers import TextStreamer

messages = [{"role": "user", "content": "為何天空是藍色的？請用繁體中文回答。"}]
try:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
except TypeError:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

print("\n[串流推理輸出]")
_ = model.generate(
    **tokenizer([text], return_tensors="pt").to("cuda"),
    max_new_tokens = 128,
    temperature = 0.7,
    top_p = 0.8,
    top_k = 20,
    streamer = TextStreamer(tokenizer, skip_prompt=True),
)

# ── 存儲模型 ──────────────────────────────────────────────────
model.save_pretrained("qwen3.5-0.8b-finetuned")
tokenizer.save_pretrained("qwen3.5-0.8b-finetuned")
print("\nModel saved to qwen3.5-0.8b-finetuned/")
