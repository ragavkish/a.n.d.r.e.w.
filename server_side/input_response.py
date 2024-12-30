from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="t5-small")

initial_prompts = ['I am A.N.D.R.E.W.!', 'hey I am A.N.D.R.E.W.!', 'hey, I am A.N.D.R.E.W.!', '. hey I am A.N.D.R.E.W.!', '. I am A.N.D.R.E.W.!', 'hi hi I am A.N.D.R.E.W.!', 'hi hi hi I am A.N.D.R.E.W.!', 'Ich bin A.N.D.R.E.W.!', 'hi hi i am A.N.D.R.E.W.!', 'hi hi I am A.N.D.R.E.W.', '! I am A.N.D.R.E.W.!', 'I am A.N.D.R.E.W.', 'I am A.N.D.R.E.W..', 'Hey I am A.N.D.R.E.W.!', 'Hey Hey I am A.N.D.R.E.W.!', 'Hey, I am A.N.D.R.E.W.!', 'Hey I am A.N.D.R.E.W.', 'Hey Hey I am A.N.D.R.E.W.', 'Hellow this is A.N.D.R.E.W.!', 'Hellow, this is A.N.D.R.E.W.!', 'Hellow This is A.N.D.R.E.W.!', 'Hellow this is A.N.D.R.E.W.', 'Rapport A.N.D.R.E.W. reporting!', 'Rapport A.N.D.R.E.W.!', 'Rapport A.N.D.R.E.W. Report!', 'Sie sprechen zu A.N.D.R.E.W.', "You're speaking to A.N.D.R.E.W.", 'A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W.', 'Sie sprechen mit A.N.D.R.E.W.', "You're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.R.E.W.", 'Heyy! A.N.D.R.E.W. hier!', 'Heyy! A.N.D.R.E.W. here!', 'Heyy, A.N.D.R.E.W. hier!', 'A.N.D.R.E.W. hier!', 'Hey, A.N.D.R.E.W. hier!', "Hey, it's A.N.D.R.E.W. here!", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W. here!", ", it's A.N.D.R.E.W. here!", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W.!", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W.", "Hello, you're chatting with A.N.D.R.E.W.", "Hello, you're chatting with A.N.D.R.E.W.!", 'Hello, you chat with A.N.D.R.E.W.', "Hello, you're talking with A.N.D.R.E.W.", 'Hello, you chat with A.N.D.R.E.W.!', 'A.N.D.R.E.W. bei Ihrem Service!', 'A.N.D.R.E.W. in Ihrem Service!', 'A.N.D.R.E.W. à votre service!', 'A.N.D.R.E.W. bei Ihnen!', 'A.N.D.R.E.W. in Ihrem service!', 'Hey, A.N.D.R.E.W. speaking!', 'Hey! A.N.D.R.E.W. speaking!', 'Hey, Hey! A.N.D.R.E.W. speaking!', 'Hey, A.N.D.R.E.W. spricht!', 'Hey, A.N.D.R.E.W. speaking! Hey!', 'Greetings, this is A.N.D.R.E.W.!', 'Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.!', 'Greetings, Greetings, this is A.N.D.R.E.W.!', 'Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.', 'Grüße, Greetings, this is A.N.D.R.E.W.!', '! A.N.D.R.E.W. reporting!', 'A.N.D.R.E.W. reporting!', 'reporting! A.N.D.R.E.W. reporting!', '! A.N.D.R.E.W. report!', 'A.N.D.R.E.W. report!', 'A.N.D.R.E.W. here!', 'A.N.D.R.E.W. here! A.N.D.R.E.W. here!', 'A.N.D.R.E.W. here! A.N.D.R.E.W.', "A.N.D.R.E.W. in the house, what's good?", "Yo! It's A.N.D.R.E.W.!", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.!", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.", "Hello, you're speaking to A.N.D.R.E.W. now!", 'Hello, you are speaking to A.N.D.R.E.W. now!', "Hello, you're speaking to A.N.D.R.E.W. today!", "Hello, you're talking to A.N.D.R.E.W. now!", "Hello, you're speaking with A.N.D.R.E.W. now!"]

trigger_inputs = ["hi", "hey", "hello", "Hello", "Hi", "Hey", "Hallo", "Hola", "Greetings", "Salutations", "Howdy", "Yo", "What's up", "Sup", "Ahoy", "Shalom", "Ciao", "Bonjour", "Aloha", "Namaste", "Salve","Konnichiwa", "G'day", "Sawasdee", "Wassup", "Hei", "Holla", "Hiyah", "Sup?", "Yo yo", "Hey there", "What's good?", "Howdy-do", "Greetings, Earthling", "Hola amigo", "Salutations, friend", "Hiya!", "Well, hello", "What's happening?", "Aloha, friend", "How's it going?", "Yo there", "Good day","Wotcha", "Peace", "Bonjourno", "What's the word?", "Hey buddy", "Howdy partner", "What’s up, fam?", "Salud", "Guten tag", "Terve", "Wassup dude", "What's cracking?", "What’s poppin’?", "Ello", "How’s life?", "Howdy, y’all", "What’s cooking?", "What’s the vibe?", "Hola, qué tal?", "Oi", "Holler", "Chao", "Que pasa?", "Wassup my dude", "Long time no see", "Heyyyy", "Yo, what’s good?", "Sup bro?", "Heyyo", "Well hey there", "G'day mate", "What's shakin'?", "How's tricks?", "Yo wassup?", "Hey friend", "How’s everything?", "Hola, amigo mío", "Bonjour, comment ça va?", "Hi, how are you?", "Oi oi", "What’s going on?", "Heeeyyy", "What’s new?", "What’s up, buddy?", "How’s your day?", "Yo fam", "Hola chicos", "Greetings, my friend", "What’s the move?", "What’s up with you?", "How's your morning?", "What’s good fam?", "Hola cómo estás?", "Hey pal", "What's up my friend?", "Hello there!", "Hey, what's new?", "What's up man?", "Yo, what’s happening?", "How’s the world treating you?", "Yo, homie", "Hey there, champ!", "How's life treating you?", "Hello sunshine", "Yo, what’s up with you?", "Hi friend!", "What's happening man?", "Hello there, stranger", "Yo, how’s it going?", "What’s good my dude?", "Hi there buddy", "Hey, what's the word?"]

target_responses = []

for prompt in initial_prompts:
    paraphrases = paraphraser(
        prompt,
        max_length=50,
        num_beams=5,
        num_return_sequences=5,
        do_sample=True,
        top_k=40,
        temperature=0.6,
    )
    for paraphrase in paraphrases:
        generated_text = paraphrase["generated_text"].strip()
        if "A.N.D.R.E.W." in generated_text and generated_text not in target_responses:
            target_responses.append(generated_text)

print("trigger_inputs =", trigger_inputs)
print("target_responses =", target_responses)
