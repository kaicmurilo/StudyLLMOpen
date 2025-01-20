import os
import pickle  # Para salvar o mapeamento `id_to_sentence`

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# Passo 1: Extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Extrai texto de cada página
    return text


# Passo 2: Dividir texto em frases
def split_into_sentences(text):
    import re

    # Divide com base em pontuação
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# Passo 3: Gerar embeddings com SentenceTransformers
def generate_embeddings(sentences, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings


# Passo 4: Criar um banco vetorial com FAISS
def create_faiss_index(embeddings, sentences):
    dimension = embeddings.shape[1]  # Tamanho do vetor
    index = faiss.IndexFlatL2(
        dimension
    )  # Cria índice FAISS baseado em L2 (distância euclidiana)
    index.add(embeddings)  # Adiciona os vetores ao índice

    # Salvar as sentenças em paralelo
    id_to_sentence = {i: sentence for i, sentence in enumerate(sentences)}
    return index, id_to_sentence


# Passo 5: Salvar banco vetorial e mapeamento
def save_faiss_index(index, id_to_sentence, index_path, mapping_path):
    # Verifica se o diretório existe; caso contrário, cria-o
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Salva o índice FAISS
    faiss.write_index(index, index_path)

    # Salva o mapeamento como um arquivo pickle
    with open(mapping_path, "wb") as f:
        pickle.dump(id_to_sentence, f)


# Passo 6: Carregar banco vetorial e mapeamento
def load_faiss_index(index_path, mapping_path):
    index = faiss.read_index(index_path)  # Carrega o índice FAISS
    with open(mapping_path, "rb") as f:
        id_to_sentence = pickle.load(f)  # Carrega o mapeamento
    return index, id_to_sentence


# Passo 7: Buscar no banco vetorial
def search_faiss(index, id_to_sentence, query, model):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(
        query_embedding, k=5
    )  # Retorna os 5 mais próximos

    # Filtrar apenas os índices válidos
    valid_results = [
        (id_to_sentence[idx], distances[0][i])
        for i, idx in enumerate(indices[0])
        if idx != -1 and idx in id_to_sentence
    ]
    return valid_results


# Exemplo de execução
if __name__ == "__main__":
    # Caminho para o PDF
    pdf_path = "pdfs/_AGRICULTURA-283 - BOLETIM SEMANAL CASA RURAL - AGRICULTURA - CIRCULAR 283.pdf"

    # Caminhos para salvar o índice e o mapeamento
    index_path = "map_local/faiss_index.bin"
    mapping_path = "map_local/id_to_sentence.pkl"

    # Verifica se o banco vetorial já existe
    if not (os.path.exists(index_path) and os.path.exists(mapping_path)):
        print("Criando novo banco vetorial...")

        # Extração e processamento
        text = extract_text_from_pdf(pdf_path)
        sentences = split_into_sentences(text)
        embeddings = generate_embeddings(sentences, model_name="all-MiniLM-L6-v2")

        # Criar e salvar banco vetorial
        index, id_to_sentence = create_faiss_index(embeddings, sentences)
        save_faiss_index(index, id_to_sentence, index_path, mapping_path)
        print("Banco vetorial criado e salvo.")
    else:
        print("Carregando banco vetorial existente...")
        index, id_to_sentence = load_faiss_index(index_path, mapping_path)

    # Consultar o banco
    print("Buscando no banco vetorial...")
    query = "Digite sua consulta aqui"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = search_faiss(index, id_to_sentence, query, model)

    # Exibir resultados
    for sentence, distance in results:
        print(f"Frase: {sentence} (Distância: {distance:.4f})")
