import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import gdown

np.random.seed(42)
torch.manual_seed(42)

tok = GPT2Tokenizer.from_pretrained("checkpoint-8000")
model = GPT2LMHeadModel.from_pretrained("checkpoint-8000")
if torch.cuda.is_available(): model.cuda()
else: model.cpu()

def set_personality():
    random_line_number = random.randint(1, 24)  # Генерация случайного номера строки

    with open('personality.txt', 'r', encoding='utf-8') as file:
        current_line_number = 0
        for line in file:
            current_line_number += 1

            if current_line_number == random_line_number:
                return line.strip()


person = set_personality()
while True:
    print("Введите запрос: ")
    prompt = input()
    if prompt == 'Кто ты?':
        print(person)
    else:
        text = '"input": ' + person + ' Продолжи диалог:\nСобеседник: ' + prompt + '\nТы: '
        inpt = tok.encode(text, return_tensors="pt")
        out = model.generate(inpt.cpu(), max_length=300, repetition_penalty=5.0, do_sample=True, top_k=5, top_p=0.95, temperature=1)
        print(tok.decode(out[0]).split('\n')[2].split('\\', 1)[0])

