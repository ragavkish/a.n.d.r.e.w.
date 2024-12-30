from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="t5-small")

initial_prompts = ["hey I am A.N.D.R.E.W.!", "hi I am A.N.D.R.E.W.!", "hello I am A.N.D.R.E.W.!", "Hey I am A.N.D.R.E.W.!", "Hellow this is A.N.D.R.E.W.!", "A.N.D.R.E.W. reporting!", "Hi! You're speaking to A.N.D.R.E.W.", "Heyy! A.N.D.R.E.W. here!"]

trigger_inputs = ["hi", "hey", "hello", "Hello", "Hi", "Hey", "Hallo", "Hola"]

target_responses = []

for prompt in initial_prompts:
    paraphrases = paraphraser(
        prompt,
        max_length=50,
        num_beams=5,
        num_return_sequences=5,
        do_sample=True,
        top_k=50,
        temperature=0.7,
    )
    for paraphrase in paraphrases:
        generated_text = paraphrase["generated_text"].strip()
        if "A.N.D.R.E.W." in generated_text and generated_text not in target_responses:
            target_responses.append(generated_text)

print("trigger_inputs =", trigger_inputs)
print("target_responses =", target_responses)
