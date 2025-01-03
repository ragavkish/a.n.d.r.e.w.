from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data = {
    "prompt": ["Hello", "Hi", "Hey", "Hallo", "Hola"],
    "response": ["Hey I am ANDREW!", "Hellow this is ANDREW!", "ANDREW reporting!", "Hi! You're speaking to ANDREW", "Heyy! ANDREW here!"],
}

dataset = Dataset.from_dict(data)

def format_data(example):
    return {"text": f"Prompt: {example['prompt']} Response: {example['response']}"}

formatted_dataset = dataset.map(format_data, remove_columns=["prompt", "response"])

train_test_split = formatted_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_dataset(dataset, tokenizer, max_length=128):
    return dataset.map(
        lambda x: {
            **tokenizer(
                x["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            ),
            "labels": tokenizer(
                x["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            )["input_ids"],
        },
        batched=True
    )

tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer)
tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer)

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="Z:/kizX/dataset/models/anderson",
    eval_strategy="steps",
    logging_dir="Z:/kizX/dataset/models/logs",
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

logger.info("Starting training...")
trainer.train()

logger.info("Saving the model and tokenizer...")
model.save_pretrained("Z:/kizX/dataset/models/anderson")
tokenizer.save_pretrained("Z:/kizX/dataset/models/anderson")

model = AutoModelForCausalLM.from_pretrained("Z:/kizX/dataset/models/anderson")
tokenizer = AutoTokenizer.from_pretrained("Z:/kizX/dataset/models/anderson")