import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Nome do modelo que deseja baixar
model_name = "gpt2"

# Diretório local para salvar o modelo
local_path = "./models/gpt2"

if not os.path.exists(local_path):
    # Baixa e salva o modelo e o tokenizer localmente
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(local_path)

# Carrega o modelo localmente
local_generator = pipeline("text-generation", model=local_path, tokenizer=local_path)


# Geração de texto usando o modelo local
def generate_text(prompt, max_length=50):
    results = local_generator(prompt, max_length=max_length, num_return_sequences=1)
    return results[0]["generated_text"]


# Teste
print(generate_text("Responde em português brasileiro. Era uma vez"))
