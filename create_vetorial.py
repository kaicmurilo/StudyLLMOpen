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


# Passo 5: Buscar no banco vetorial
def search_faiss(index, id_to_sentence, query, model):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(
        query_embedding, k=5
    )  # Retorna os 5 mais próximos

    # Mostrar resultados
    results = [(id_to_sentence[idx], distances[i]) for i, idx in enumerate(indices[0])]
    return results


# Exemplo de execução
if __name__ == "__main__":
    # Caminho para o PDF
    pdf_path = "pdfs/_AGRICULTURA-283 - BOLETIM SEMANAL CASA RURAL - AGRICULTURA - CIRCULAR 283.pdf.pdf"

    # Extração e processamento
    print("Extraindo texto do PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Dividindo texto em frases...")
    sentences = split_into_sentences(text)

    print(f"Gerando embeddings para {len(sentences)} frases...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = generate_embeddings(sentences, model_name="all-MiniLM-L6-v2")

    print("Criando banco vetorial com FAISS...")
    index, id_to_sentence = create_faiss_index(embeddings, sentences)

    # Consultar o banco
    print("Buscando no banco vetorial...")
    query = "Digite sua consulta aqui"
    results = search_faiss(index, id_to_sentence, query, model)

    # Exibir resultados
    for sentence, distance in results:
        print(f"Frase: {sentence} (Distância: {distance:.4f})")
