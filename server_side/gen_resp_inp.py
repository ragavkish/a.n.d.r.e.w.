import random
from nltk.corpus import wordnet

import nltk
nltk.download('wordnet')

def generate_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def generate_trigger_inputs(base_inputs, num_variants=8):
    new_inputs = []
    for trigger in base_inputs:
        new_inputs.append(trigger)
        synonyms = generate_synonyms(trigger)
        random_synonym = random.choice(synonyms) if synonyms else trigger
        new_inputs.append(random_synonym)
    return random.sample(new_inputs, num_variants)

def generate_target_responses(base_responses, num_variants=8, skip_words=None):
    if skip_words is None:
        skip_words = []

    new_responses = []
    for response in base_responses:
        new_responses.append(response)
        words = response.split()
        for i in range(len(words)):
            if words[i] in skip_words:
                continue
            synonyms = generate_synonyms(words[i])
            if synonyms:
                word_variant = random.choice(synonyms)
                words[i] = word_variant
        new_responses.append(' '.join(words))
    return random.sample(new_responses, num_variants)

trigger_inputs = ["hi", "hey", "hello", "Hello", "Hi", "Hey", "Hallo", "Hola"]
target_responses = ["hey I am ANDREW!", "hi I am ANDREW!", "hello I am ANDREW!", 
                    "Hey I am ANDREW!", "Hellow this is ANDREW!", "ANDREW reporting!", 
                    "Hi! You're speaking to ANDREW", "Heyy! ANDREW here!"]

skip_list = ["ANDREW"]

generated_inputs = generate_trigger_inputs(trigger_inputs)
generated_responses = generate_target_responses(target_responses, skip_words=skip_list)

print("Generated Trigger Inputs:")
for input_text in generated_inputs:
    print(input_text)

print("\nGenerated Target Responses:")
for response_text in generated_responses:
    print(response_text)
