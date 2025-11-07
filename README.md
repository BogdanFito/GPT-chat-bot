# Проект обучения llm-моделей на датасете из статьи Habr 
[ссылка на статью](https://habr.com/ru/articles/751580/)

## Обученные модели:
1. [RuGPT3Small](https://colab.research.google.com/github/ai-forever/ru-gpts/blob/master/examples/RuGPT3FinetuneHF.ipynb)

Самая лучшая версия - checkpoint-240000

Обучение происходило без валидации с дефолтными параметрами на видеокарте NVIDIA GeForce RTX 4070 Ti SUPER

Для старта обучения необходимо установить нужные библиотеки и выполнить скрипт 

```
    python run_clm.py --model_name_or_path sberbank-ai/rugpt3small_based_on_gpt2 --train_file dataset.txt --per_device_train_batch_size 1 --block_size 2048 --dataset_config_name plain_text --do_train --output_dir models/gpt --save_steps 80000 --num_train_epochs 
```