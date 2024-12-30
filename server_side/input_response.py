from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="t5-small")

prompts = ["Hello", "Hi", "Hey there!"]
expanded_prompts = []

for prompt in prompts:
    paraphrases = paraphraser(
        f"{prompt}",
        max_length=50,
        num_beams=5,             
        num_return_sequences=5,  
        do_sample=True,          
        top_k=50,                
        temperature=0.7,         
    )
    expanded_prompts.extend([p["generated_text"] for p in paraphrases])

print(expanded_prompts)
