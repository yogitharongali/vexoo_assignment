from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("openai/gsm8k", "main")

train_data = dataset["train"].select(range(3000))
test_data = dataset["test"].select(range(1000))

# -----------------------------
# Model (use GPT-2 instead of LLaMA)
# -----------------------------
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

#  Add padding token
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Tokenization
# -----------------------------
def tokenize(example):
    encoding = tokenizer(
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    #  Add labels for loss
    encoding["labels"] = encoding["input_ids"].copy()
    
    return encoding

train_data = train_data.map(tokenize)
test_data = test_data.map(tokenize)

# -----------------------------
# Model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)

# -----------------------------
# LoRA Config (fixed for GPT-2)
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # ✅ FIX 3
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=100
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Evaluate
# -----------------------------
results = trainer.evaluate()
print(results)
