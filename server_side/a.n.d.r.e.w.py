from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset

data = {
    "prompt": ["Hello, who are you?", "What's the weather?"],
    "response": ["I'm your assistant, here to help!", "It's sunny with a high of 75°F."],
}

dataset = Dataset.from_dict(data)

def format_data(example):
    return {"text": f"{example['prompt']} {example['response']}"}

formatted_dataset = dataset.map(format_data, remove_columns=["prompt", "response"])

train_test_split = formatted_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_data_with_labels(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize_data_with_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_data_with_labels, batched=True)

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="models/anderson",
    eval_strategy="steps",
    logging_dir="models/logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./models/anderson")
tokenizer.save_pretrained("./models/anderson")

model = AutoModelForCausalLM.from_pretrained("./models/anderson")
tokenizer = AutoTokenizer.from_pretrained("./models/anderson")