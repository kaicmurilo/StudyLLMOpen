import os

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Nome do modelo e diretório local
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Substitua pelo modelo desejado
local_path = "./models/Llama-3.2-1B-Instruct"

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
model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto")

# Configura o pipeline para geração de texto
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt fixo
fixed_prompt = "Você é um assistente amigável que responde perguntas de maneira educada e informativa. Não criar conversas, apenas responder a pergunta. Não retornar o histórico da conversa."


# Função para gerar respostas
def respond(message, history):
    prompt = f"{fixed_prompt}\n\nPergunta: {message}\n\nResposta:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"].split("Resposta:")[1].strip()


# Interface Gradio com ChatInterface
demo = gr.ChatInterface(
    respond,
    title="Chat com LLaMA",
    description="Um assistente amigável que responde perguntas com informações úteis e claras.",
)

# Inicia o servidor Gradio
demo.launch()
