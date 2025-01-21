import os
import pickle

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Caminhos do banco vetorial
index_path = "map_local/faiss_index.bin"
mapping_path = "map_local/id_to_sentence.pkl"

# Nome do modelo e diretório local
model_name = "meta-llama/Llama-3.2-1B-Instruct"
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
fixed_prompt = "Você é um assistente especializado em fornecer previsões, análises, insights e dados relevantes de maneira objetiva e direta. Responda com clareza e precisão, mantendo as respostas concisas e limitadas a 2000 tokens. Evite redundâncias e não inclua o contexto na resposta. Foque exclusivamente em atender à pergunta com informações assertivas e relevantes."


# Funções para trabalhar com o banco vetorial
def load_faiss_index(index_path, mapping_path):
    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        id_to_sentence = pickle.load(f)
    return index, id_to_sentence


def search_faiss(index, id_to_sentence, query, model):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=20)
    valid_results = [
        id_to_sentence[idx] for idx in indices[0] if idx != -1 and idx in id_to_sentence
    ]
    return "\n".join(valid_results)


# Carrega o banco vetorial
if os.path.exists(index_path) and os.path.exists(mapping_path):
    print("Carregando banco vetorial...")
    index, id_to_sentence = load_faiss_index(index_path, mapping_path)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
else:
    raise RuntimeError(
        "Banco vetorial não encontrado. Certifique-se de criá-lo antes de usar este script."
    )


# Função para gerar respostas com base no contexto
def respond(message, history):
    # Busca contexto relevante no banco vetorial
    context = search_faiss(index, id_to_sentence, message, embedding_model)

    # Gera a resposta do modelo usando o contexto
    prompt = (
        f"{fixed_prompt}\n\nContexto:\n{context}\n\nPergunta: {message}\n\nResposta:"
    )
    response = generator(
        prompt, max_new_tokens=2048, return_full_text=False, num_return_sequences=1
    )
    return response[0]["generated_text"]


# Interface Gradio com ChatInterface
demo = gr.ChatInterface(
    respond,
    title="Chat com LLaMA usando Base Vetorial",
    description="Um assistente amigável que usa um banco vetorial para fornecer respostas com base em contexto relevante.",
)

# Inicia o servidor Gradio
demo.launch()
