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

trigger_inputs = ['hi', 'hey', 'hello', 'Hello', 'Hi', 'Hey', 'Hallo', 'Hola', 'Greetings', 'Salutations', 'Howdy', 'Yo', "What's up", 'Sup', 'Ahoy', 'Shalom', 'Ciao', 'Bonjour', 'Aloha', 'Namaste', 'Salve', 'Konnichiwa', "G'day", 'Sawasdee', 'Wassup', 'Hei', 'Holla', 'Hiyah', 'Sup?', 'Yo yo', 'Hey there', "What's good?", 'Howdy-do', 'Greetings, Earthling', 'Hola amigo', 'Salutations, friend', 'Hiya!', 'Well, hello', "What's happening?", 'Aloha, friend', "How's it going?", 'Yo there', 'Good day', 'Wotcha', 'Peace', 'Bonjourno', "What's the word?", 'Hey buddy', 'Howdy partner', 'What’s up, fam?', 'Salud', 'Guten tag', 'Terve', 'Wassup dude', "What's cracking?", 'What’s poppin’?', 'Ello', 'How’s life?', 'Howdy, y’all', 'What’s cooking?', 'What’s the vibe?', 'Hola, qué tal?', 'Oi', 'Holler', 'Chao', 'Que pasa?', 'Wassup my dude', 'Long time no see', 'Heyyyy', 'Yo, what’s good?', 'Sup bro?', 'Heyyo', 'Well hey there', "G'day mate", "What's shakin'?", "How's tricks?", 'Yo wassup?', 'Hey friend', 'How’s everything?', 'Hola, amigo mío', 'Bonjour, comment ça va?', 'Hi, how are you?', 'Oi oi', 'What’s going on?', 'Heeeyyy', 'What’s new?', 'What’s up, buddy?', 'How’s your day?', 'Yo fam', 'Hola chicos', 'Greetings, my friend', 'What’s the move?', 'What’s up with you?', "How's your morning?", 'What’s good fam?', 'Hola cómo estás?', 'Hey pal', "What's up my friend?", 'Hello there!', "Hey, what's new?", "What's up man?", 'Yo, what’s happening?', 'How’s the world treating you?', 'Yo, homie', 'Hey there, champ!', "How's life treating you?", 'Hello sunshine', 'Yo, what’s up with you?', 'Hi friend!', "What's happening man?", 'Hello there, stranger', 'Yo, how’s it going?', 'What’s good my dude?', 'Hi there buddy', "Hey, what's the word?"]
target_responses = ['Ich bin A.N.D.R.E.W.!', 'I am A.N.D.R.E.W.!', 'Ich bin A.N.D.R.E.W.', 'hey I am A.N.D.R.E.W.!', 'hey, I am A.N.D.R.E.W.!', '. I am A.N.D.R.E.W.!', ', I am A.N.D.R.E.W.!', 'hey, hey, I am A.N.D.R.E.W.!', ', hey, I am A.N.D.R.E.W.!', 'hey hey I am A.N.D.R.E.W.!', 'hey I am A.N.D.R.E.W.', 'A.N.D.R.E.W.!', 'Am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.!', 'Mein bin A.N.D.R.E.W.!', 'ich bin A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi i am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi i am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi hi i am A.N.D.R.E.W.!', 'hi hi hi hi hi hi i am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi hi hi hi i am A.N.D.R.E.W.!', 'hi hi hi hi hi hi hi I am A.N.D.R.E.W.', 'hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.', 'hi hi hi hi hi hi I am A.N.D.R.E.W.', 'hi hi hi hi hi I am A.N.D.R.E.W.', 'hi hi hi hi hi hi hi hi hi hi I am A.N.D.R.E.W.', '! I am A.N.D.R.E.W.!', 'Ich bin A.N.D.R.E.W. !', 'I am A.N.D.R.E.W.', 'Ich bin A.N.D.R.E.W.A.', 'Ich bin A.N.D.R.E.W..', 'Hey I am A.N.D.R.E.W.!', 'Hey Hey I am A.N.D.R.E.W.!', 'Hey, I am A.N.D.R.E.W.!', 'Hey I am A.N.D.R.E.W.', 'Hey Hey I am A.N.D.R.E.W.', 'Hey Hey Hey I am A.N.D.R.E.W.!', 'Hey Hey Hey Hey I am A.N.D.R.E.W.!', 'Hey Hey Hey, I am A.N.D.R.E.W.!', 'Hey Hey, I am A.N.D.R.E.W.!', 'Hey, I am A.N.D.R.E.W.', 'Hey, Hey, I am A.N.D.R.E.W.!', 'Hey, Hey, I am A.N.D.R.E.W.', 'Hey Hey Hey I am A.N.D.R.E.W.', 'Hey Hey Hey Hey I am A.N.D.R.E.W.', 'Hey Hey Hey, I am A.N.D.R.E.W.', 'Hey Hey Hey Hey Hey I am A.N.D.R.E.W.', 'Hellow this is A.N.D.R.E.W.!', 'Hellow, this is A.N.D.R.E.W.!', 'Hellow This is A.N.D.R.E.W.!', 'Hellow this is A.N.D.R.E.W.', 'Hellow, this is A.N.D.R.E.W.', 'Hellow, This is A.N.D.R.E.W.!', 'Hellow, it is A.N.D.R.E.W.!', 'Hellow Hellow This is A.N.D.R.E.W.!', 'Hellow Hellow Hellow This is A.N.D.R.E.W.!', 'Hellow Hellow this is A.N.D.R.E.W.', 'Hellow This is A.N.D.R.E.W.', 'Hellow, Hellow this is A.N.D.R.E.W.', 'Rapport A.N.D.R.E.W. reporting!', 'Rapport Rapport A.N.D.R.E.W. reporting!', 'Rapport A.N.D.R.E.W. report!', 'Rapport A.N.D.R.E.W.!', 'Rapport Rapport A.N.D.R.E.W.!', 'Rapport A.N.D.R.E.W.', 'Rapport A.N.D.R.E.W..', 'Rapport A.N.D.R.E.W. Rapport!', 'Rapport A.N.D. Rapport A.N.D.R.E.W. Rapport!', 'Rapport A.N.D.R.E.W. Rapport A.N.D. Rapport A.N.D.R.E.W.', 'Rapport A.N.D.R.E.W. Report!', 'Rapport A.N.D. Rapport A.N.D.R.E.W.', 'Sie sprechen zu A.N.D.R.E.W.', 'Sie sprechen bei A.N.D.R.E.W.', 'Sie sprechen mit A.N.D.R.E.W.', 'Sie sprechen zu A.N.D.R.E.W., A.N.D.R.E.W.', 'Sie sprechen zu A.N.D.R.E.W., A.N.D.D.R.E.W.', 'Sie sprechen to A.N.D.R.E.W.', 'Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A', 'Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W.', 'Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W.', 'Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W.Sie sprechen zu A', 'Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A.N.D.R.E.W.Sie sprechen zu A.N.D.R.E.W. Sie sprechen zu A', 'Sie sprechen mit A.N.D.R.E.W. Sie sprechen mit A.N.D.R.E.W.', 'Sie sprechen mit A.N.D.R.E.W., Sie sprechen mit A.N.D.R.E.W.', 'Wenn Sie mit A.N.D.R.E.W. sprechen, sprechen Sie mit A.N.D.R.E.W.', "You're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.", "Sie're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.", "Sie sprechen to A.N.D.R.E.W., and you're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.R.", "You're speaking to A.N.D.R.E.W. and you're speaking to A.N.D.R.E.W., and you're speaking to A.N.D.R", 'Sie sprechen zu A.N.D.R.E.W., und Sie sprechen zu A.N.D.R.E.W., und Sie sprechen zu A.N.D.R.E.W.', 'Hier! A.N.D.R.E.W. hier!', 'Heyy! A.N.D.R.E.W. hier!', 'Hier! A.N.D.R.E.W. Hier!', 'Heyy! A.N.D.R.E.W. here!', 'Heyy, A.N.D.R.E.W. hier!', 'A.N.D.R.E.W. hier!', 'Hey, A.N.D.R.E.W. hier!', 'Hier! Heyy, A.N.D.R.E.W. hier!', 'hier! Heyy, A.N.D.R.E.W. hier!', 'Hier! Heyy, A.N.D.R.E.W. Hier!', 'Hier! Hey, A.N.D.R.E.W. hier!', 'Hier, Hey, A.N.D.R.E.W. hier!', 'Hey, hier, A.N.D.R.E.W. hier!', 'Hey, hier, A.N.D.R.E.W.', "Hey, it's A.N.D.R.E.W. here!", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W. here!", ", it's A.N.D.R.E.W. here!", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W.!", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W.", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.", "Hey, it's A.N.D.R.E.W. here! Hey, Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W. here! Hey, Hey, it's A.N.", "Hey, Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.", ', es ist A.N.D.R.E.W. hier!', ', es ist A.N.D.R.E.W. here!', 'Es ist A.N.D.R.E.W. hier!', "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W.! Hey, it's A.N.D.R", "Hey, it's A.N.D.R.E.W.! Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R", "Hey, it's A.N.D.R.E.W here! Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R", "Hey, it's A.N.D.R.E.W. here! Hey, it's A.N.D.R.E.W. here! Hey, it's a.N.D", "Hello, you're chatting with A.N.D.R.E.W.", 'Hello, you chat with A.N.D.R.E.W.', "Hello, you're talking with A.N.D.R.E.W.", "Hello, you're chatting to A.N.D.R.E.W.", "Hello, you're chatting with A.N.D.R.E.W.!", 'Hello, you chat with A.N.D.R.E.W.!', 'Hallo, Sie chatte mit A.N.D.R.E.W.', 'Hallo, du chatst mit A.N.D.R.E.W.', 'Hallo, du chattst mit A.N.D.R.E.W.', "Hello, you're talking with A.N.D.R.E.W. Hello, you're talking with A.N.D.R.E.W.", "Hello, you're talking with A.N.D.R.E.W., you're talking with A.N.D.R.E.W.", "Hello, you're talking to A.N.D.R.E.W.", "Hello, you're talking with A.N.D.R.E.W., and you're talking with A.N.D.R.E.W.", 'Hello Hello, you chat with A.N.D.R.E.W.', 'Hallo, chatte mit A.N.D.R.E.W.', 'A.N.D.R.E.W. bei Ihrem Service!', 'Bei Ihrem Service: A.N.D.R.E.W. bei Ihrem Service!', 'A.N.D.R.E.W. in Ihrem Service!', 'A.N.D.R.E.W. à votre service!', 'A.N.D.R.E.W. à vos service!', 'A.N.D.R.E.W. à vos services!', 'A.N.D.R.E.W. bei Ihnen!', 'A.N.D.R.E.W. bei dir!', 'A.N.D.R.E.W. in Ihrem service!', 'Hey, A.N.D.R.E.W. speaking!', 'Hey, A.N.D.R.E.W. sprechen!', 'Hey, A.N.D.R.E.W. spricht!', 'Hey, A.N.D.R.E.W. speaks!', 'Hey, A.N.D.R.E.W., spricht!', 'Hey, Hey! A.N.D.R.E.W. speaking!', 'Hey! Hey! A.N.D.R.E.W. speaking!', 'Hey, Hey! A.N.D.R.E.W. spricht!', 'Hey, Hey! A.N.D.R.E.W. speaking! Hey!', 'Hey! Hey! A.N.D.R.E.W. spricht!', 'Hey, Hey, Hey! A.N.D.R.E.W. spricht!', 'Hey, Hey! A.N.D.R.E.W.', 'Hey, Hey, Hey! A.N.D.R.E.W.', 'Hey, A.N.D.R.E.W.', 'Hey A.N.D.R.E.W. spricht!', 'Hey Hey, A.N.D.R.E.W. spricht!', 'Hey, A.N.D.R.E.W. speaking! Hey!', 'A.N.D.R.E.W. speaking! Hey!', 'Hey A.N.D.R.E.W. speaking!', 'Greetings, this is A.N.D.R.E.W.!', 'Grüße, Greetings, this is A.N.D.R.E.W.!', 'Greetings, this is A.N.D.R.E.W.!!', 'Greetings, This is A.N.D.R.E.W.!', 'Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D', 'Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.', 'Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.!', 'This is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E', 'Greetings, Greetings, this is A.N.D.R.E.W.!', 'Greetings, Greetings, this is A.N.D.R.E.W.!!', 'Greetings Greetings, Greetings, this is A.N.D.R.E.W.!', '! Greetings, Greetings, this is A.N.D.R.E.W.!', '! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N', ', this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R', '! Greetings, this is A.N.D.R.E.W.! Greetings, this is A.N.D.R.E.W.!', 'Grüße, Grüße, Greetings, this is A.N.D.R.E.W.!', 'Grüße, Grüße, Grüße, Greetings, this is A.N.D.R.E.W.!', 'Grüße, Grüße, Grüße, Grüße, Greetings, this is A.N.D.R.E.W.!', 'A.N.D.R.E.W. reporting!', '! A.N.D.R.E.W. reporting!', 'A.N.D.R.E.W. report!', 'Rapport A.N.D.R.E.W. !', 'A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting!', 'A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting! A.N.D.R', 'Reporting! A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting!', 'A.N.D.R.E.W. reporting! A.N.D.R.E.W. reporting!', 'Reporting! A.N.D.R.E.W. Reporting! A.N.D.R.E.W. Reporting! A.N.D.R.E.W. Reporting!', 'Bericht A.N.D.R.E.W.', 'A.N.D.R.E.W.', 'Der Bericht A.N.D.R.E.W.', 'A.N.D.R.E.W. here!', '! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D', '! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here!', 'here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.', '! Here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N', '! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! Here! A.N', 'A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R', 'here! A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here!', 'A.N.D.R.E.W. here! A.N.D.R.E.W. here! A.N.D.R.E.W. here!', 'here! A.N.D.R.E.W. here! A.N.D.R.E.W. here!', "A.N.D.R.E.W. in the house, what's good?", "Yo! It's A.N.D.R.E.W.!", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.!", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! it's A.N.D.R.E.W.", "It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.!", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.W", "Yo! Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.", "Yo Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.W", "Yo! it's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.", "Yo! It's A.N.D.R.E.W.! It's A.N.D.R.E.W.! It's A.n.D.R.E.W.", "Hello, you're speaking to A.N.D.R.E.W. now!", 'Hello, you are speaking to A.N.D.R.E.W. now!', "Hello, you're speaking to A.N.D.R.E.W. today!", "Hello, you're talking to A.N.D.R.E.W. now!", "Hello, you're speaking with A.N.D.R.E.W. now!", 'Hello, you speak to A.N.D.R.E.W. now!', 'Hallo, Sie sprechen jetzt zu A.N.D.R.E.W.', 'Hello, you are speaking to A.N.D.R.E.W. today!', 'Hallo, Sie sprechen jetzt mit A.N.D.R.E.W.', "Hello, you're speaking to A.N.D.R.E.W.", "Hello, you're speaking today to A.N.D.R.E.W.", "Hello, you're speaking with A.N.D.R.E.W. today!", "Hello, you're talking with A.N.D.R.E.W. now!", 'Hello, you are talking to A.N.D.R.E.W. now!', "Hello, you're talking to A.N.D.R.E.W. today!", 'Hello, you are speaking with A.N.D.R.E.W. now!', "Hello, you're speaking with A.N.D.R.E.W."]

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