import json
import os
from difflib import get_close_matches

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Caminhos
MODEL_NAME = "openlm-research/open_llama_3b_v2"
LOCAL_MODEL_DIR = "./local_openllama_model"
DOCUMENTS_FILE = "agricultura_docs.json"

# Configurar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Verificar e carregar modelo local
if not os.path.exists(LOCAL_MODEL_DIR):
    print("Modelo local não encontrado. Baixando...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("Modelo baixado e salvo localmente.")
else:
    print("Carregando modelo local...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)

model = model.to(device)
print("Modelo carregado com sucesso.")


# Função para carregar os documentos indexados
def load_documents():
    """Carrega os documentos indexados do arquivo JSON."""
    if not os.path.exists(DOCUMENTS_FILE):
        raise FileNotFoundError("Nenhum documento foi indexado ainda.")
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Função para buscar documentos relevantes
def search_documents(query: str, top_k=1):
    """Busca documentos relevantes para a pergunta no JSON indexado."""
    documents = load_documents()
    titles = [doc["title"] for doc in documents]
    matched_titles = get_close_matches(query, titles, n=top_k, cutoff=0.1)
    return [doc for doc in documents if doc["title"] in matched_titles]


# Função para gerar uma resposta
def respond(message, history, system_message, max_tokens, temperature, top_p):
    """Gera uma resposta combinando a pergunta com o contexto relevante."""
    # Carregar o contexto
    relevant_docs = search_documents(message, top_k=1)
    if not relevant_docs:
        yield "Nenhum contexto relevante encontrado para a pergunta."
        return

    context = relevant_docs[0]["content"]

    # Formatar o prompt com o contexto e a pergunta
    prompt = f"Você é um assistente virtual especializado em informações agrícolas.\n\nContexto: {context}\n\nPergunta: {message}\nResposta:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    ).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    yield response


# Interface do Gradio
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="Você é um assistente virtual amigável especializado em informações agrícolas. Responda com base no contexto fornecido.",
            label="System message",
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

# Iniciar a interface
if __name__ == "__main__":
    demo.launch()
