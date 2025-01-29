import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# Nome do modelo e diretório local
model_name = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Alterar para outro se necessário
)
local_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B"

# Verifica se o modelo já foi baixado
if not os.path.exists(local_path):
    print("Modelo não encontrado localmente. Baixando...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.save_pretrained(local_path)
else:
    print("Modelo encontrado localmente. Carregando...")

# Carrega o modelo local
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path, device_map="auto"  # Carregamento leve
)

# Geração de texto
from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Teste simples
prompt = "Explique os benefícios da energia solar:"
response = generator(prompt, max_length=100, num_return_sequences=1)
print("Resposta gerada:")
print(response[0]["generated_text"])
