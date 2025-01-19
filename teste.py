from transformers import AutoModelForCausalLM, AutoTokenizer

# Carregar o modelo ajustado
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_llama")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llama")

# Gerar texto a partir de um prompt
prompt = "Qual foi a produtividade da soja na safra 2019/2020?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
