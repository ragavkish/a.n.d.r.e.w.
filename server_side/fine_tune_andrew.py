import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import gc
import os
import shutil

model = AutoModelForCausalLM.from_pretrained("Z:/kizX/dataset/models/anderson")
tokenizer = AutoTokenizer.from_pretrained("Z:/kizX/dataset/models/anderson")

def fine_tune_model(new_data):
    dataset = Dataset.from_dict(new_data)
    formatted_dataset = dataset.map(
        lambda x: {"text": f"{x['prompt']} {x['response']}"},
        remove_columns=["prompt", "response"]
    )
    
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128),
        batched=True
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"labels": x["input_ids"].copy()}
    )
    
    training_args = TrainingArguments(
        output_dir="Z:/kizX/dataset/models/anderson_temp",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    model.cpu()

    temp_dir = "Z:/kizX/dataset/models/anderson_temp"
    save_dir = "Z:/kizX/dataset/models/anderson"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    shutil.move(temp_dir, save_dir)

    print(f"Model saved to {save_dir}")

target_responses = ["hey I am ANDREW!", "hi I am ANDREW!", "hello I am ANDREW!"]
trigger_inputs = ["hi", "hey", "hello"]

while True:
    user_input = trigger_inputs[0]
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Model: {response}")
    
    if response.strip().lower() in target_responses:
        print("Success: Model returned the expected response!")
        break
    
    print("Fine-tuning the model...")
    
    new_data = {
        "prompt": [user_input],
        "response": [target_responses[0]]
    }
    fine_tune_model(new_data)
    print("Model fine-tuned with the new example!")
