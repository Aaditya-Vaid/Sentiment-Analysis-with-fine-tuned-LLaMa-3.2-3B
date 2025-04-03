# Sentiment-Analysis-with-fine-tuned-LLaMa-3.2-3B
This project aims at building a Machine Learning model for Sentiment Analysis task using a fine-tuned LLM.
I used LLaMa-3.2-3B model because of its less number of parameters. Also used unsloth for faster fine tuning.

## Data scraping for training
Used python and selenium to build a web-scrapper to collect product reviews in their original form. Collected 8000+ reviews.
Utilized amazoncaptcha to bypass captcha page of amazon.

```python
from amazoncaptcha import AmazonCaptcha
``` 

You can easily install amazoncaptcha using following pip command in the notebook

```notebook
!pip install amazoncaptcha
```

Saved reviews in a ```.csv``` format.

## Data Labeling
Labeled the reviews as positive or negative using transformer pipeline for sentiment analysis task to ensure accurate labeling of reviews.

```python
from transformers import pipeline
inport torch
device = 0 if torch.cuda.is_available() else -1
model = pipeline("sentiment-analysis",
                 model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                 tokenizer="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                 truncation=True,
                 device = device)
```

## Unsloth installation
```notebook
%%capture

!pip install unsloth # install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git # Also get the latest version Unsloth!
```
