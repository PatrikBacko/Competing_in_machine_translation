# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


#import model from local directory .\opus-mt-cs-en
tokenizer = AutoTokenizer.from_pretrained("Competing_in_machine_translation\opus-mt-cs-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Competing_in_machine_translation\opus-mt-cs-en")


# Translate Czech to English
text = "Dobrý den, jak se máte?"
encoded = tokenizer.encode(text, return_tensors="pt") 
generated_tokens = model.generate(encoded, num_beams=5, num_return_sequences=3)
result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
for i, r in enumerate(result):
    print(f"{i+1}. {r}")