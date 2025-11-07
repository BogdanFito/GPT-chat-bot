import random

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

np.random.seed(42)
torch.manual_seed(42)

tok = GPT2Tokenizer.from_pretrained("models/gpt/checkpoint-320000")
model = GPT2LMHeadModel.from_pretrained("models/gpt/checkpoint-320000")
model.cuda()
person_list = []
with open('personality.txt','r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if line != '':
            person_list.append(line)
        else:
            break

while True:
    text = input('Введите запрос:')
    result = ''
    while len(result)<5:
        person = person_list[random.randint(0,len(person_list)-1)]
        prompt = f'<SC6>{person} Продолжи диалог:' + text.strip() + '\nТы: <extra_id_0>'
        inpt = tok.encode(prompt, return_tensors="pt")
        out = model.generate(inpt.cuda(), max_length=500, repetition_penalty=1.2, do_sample=True, top_k=2, top_p=0.85,
                             temperature=0.9)
        result = tok.decode(out[0], skip_special_tokens=True)
        result = result.split('\n')[2]
        while '\\n' in result:
            result = result.split('\\n')[0]
        result = result.replace('Ты: <extra_id_0>', '')
        result = result.replace('"output":', '')
        result = result.replace('Собеседник:', '')

    print(result)