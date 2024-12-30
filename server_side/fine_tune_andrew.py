import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import gc
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = AutoModelForCausalLM.from_pretrained("Z:/kizX/dataset/models/anderson")
tokenizer = AutoTokenizer.from_pretrained("Z:/kizX/dataset/models/anderson")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def is_response_valid(response, target_responses, threshold=0.8):
    response_embedding = similarity_model.encode(response)
    target_embeddings = similarity_model.encode(target_responses)
    cosine_scores = util.pytorch_cos_sim(response_embedding, target_embeddings)
    return cosine_scores.max().item() > threshold

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
    
    logger.info("Fine-tuning the model...")
    trainer.train()
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    model.cpu()

    temp_dir = "Z:/kizX/dataset/models/anderson_temp"
    save_dir = "Z:/kizX/dataset/models/anderson"

    if os.path.exists(save_dir):
        logger.info(f"Removing existing directory at {save_dir}...")
        shutil.rmtree(save_dir)

    logger.info(f"Moving {temp_dir} to {save_dir}...")
    shutil.move(temp_dir, save_dir)
    logger.info(f"Model saved to {save_dir}")

target_responses = ["hey I am ANDREW!", "hi I am ANDREW!", "hello I am ANDREW!", "Hey I am ANDREW!", "Hellow this is ANDREW!", "ANDREW reporting!", "Hi! You're speaking to ANDREW", "Heyy! ANDREW here!"]
trigger_inputs = ["hi", "hey", "hello", "Hello", "Hi", "Hey", "Hallo", "Hola"]

max_iterations = 5
iteration = 0

while iteration < max_iterations:
    user_input = trigger_inputs[iteration % len(trigger_inputs)]
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Model response: {response}")

    if is_response_valid(response, target_responses):
        logger.info("Success: Model returned the expected response!")
        break

    logger.info("Fine-tuning the model with new data...")
    new_data = {"prompt": [user_input], "response": [target_responses[0]]}
    fine_tune_model(new_data)
    iteration += 1

if iteration == max_iterations:
    logger.warning("Model failed to converge after maximum iterations.")
