import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

model = AutoModelForCausalLM.from_pretrained("./models/anderson")
tokenizer = AutoTokenizer.from_pretrained("./models/anderson")

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
        output_dir="Z:/kizX/dataset/models/anderson",
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
    model.save_pretrained("Z:/kizX/dataset/models/anderson")
    tokenizer.save_pretrained("Z:/kizX/dataset/models/anderson")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chat.")
        break

    expected_output = input("Expected Output: ")
    
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Model: {response}")
    
    fine_tune = input("Fine-tune with this example? (yes/no): ").lower()
    if fine_tune == "yes":
        new_data = {
            "prompt": [user_input],
            "response": [expected_output]
        }
        fine_tune_model(new_data)
        print("Model fine-tuned with the new example!")
