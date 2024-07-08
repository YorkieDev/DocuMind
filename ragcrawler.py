import streamlit as st
import os
import tempfile
import docx2txt
import PyPDF2
from openai import OpenAI
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Any
import tiktoken
import pickle
from streamlit_ace import st_ace
import latex2mathml
import requests
import re
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Streamlit configuration
st.set_page_config(page_title="DocuMind", layout="wide")

# OpenAI client configuration
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Constants
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
CHAT_MODEL = "lmstudio-community/gemma-2-27b-it-GGUF"
TOKENIZER = tiktoken.get_encoding("cl100k_base")

class DocumentProcessor:
    @staticmethod
    def process(file) -> str:
        file_extension = os.path.splitext(file.name)[1].lower()
        processors = {
            ".txt": lambda f: f.getvalue().decode("utf-8"),
            ".docx": docx2txt.process,
            ".pdf": lambda f: " ".join(page.extract_text() for page in PyPDF2.PdfReader(f).pages)
        }
        processor = processors.get(file_extension)
        if not processor:
            raise ValueError(f"Unsupported file format: {file_extension}")
        return processor(file)

    @staticmethod
    def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

class OpenAIWrapper:
    @staticmethod
    def get_embedding(text: str) -> List[float]:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

    @staticmethod
    def chat_completion(messages: List[Dict[str, str]], model: str = CHAT_MODEL) -> str:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content

class RAGSystem:
    def __init__(self):
        self.documents: List[str] = []
        self.index: Any = None
        self.vectorizer = TfidfVectorizer()

    def add_document(self, content: str):
        chunks = DocumentProcessor.split_text(content)
        self.documents.extend(chunks)
        self._update_index()

    def _update_index(self):
        if self.documents:
            embeddings = [OpenAIWrapper.get_embedding(doc) for doc in self.documents]
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(np.array(embeddings))

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        if not self.documents or self.index is None:
            return []
        query_embedding = OpenAIWrapper.get_embedding(query)
        _, indices = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in indices[0]]

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump((self.documents, self.index, self.vectorizer), f)

    @classmethod
    def load(cls, filename: str):
        rag = cls()
        with open(filename, 'rb') as f:
            rag.documents, rag.index, rag.vectorizer = pickle.load(f)
        return rag

    def clear(self):
        self.documents = []
        self.index = None
        self.vectorizer = TfidfVectorizer()

class WebCrawler:
    @staticmethod
    def crawl(url: str, max_pages: int = 5) -> str:
        visited = set()
        to_visit = [url]
        all_text = []

        def is_valid_url(url):
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)

        def process_url(current_url):
            try:
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()
                text = WebCrawler.extract_text_from_html(response.text)
                links = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
                return current_url, text, links
            except Exception as e:
                print(f"Error crawling {current_url}: {str(e)}")
                return current_url, "", []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            while to_visit and len(visited) < max_pages:
                current_url = to_visit.pop(0)
                if current_url not in visited:
                    visited.add(current_url)
                    futures.append(executor.submit(process_url, current_url))

            for future in as_completed(futures):
                current_url, text, links = future.result()
                if text:
                    all_text.append(f"URL: {current_url}\n\n{text}\n\n{'='*50}\n")
                for link in links:
                    absolute_link = urljoin(current_url, link)
                    if is_valid_url(absolute_link) and urlparse(absolute_link).netloc == urlparse(url).netloc:
                        to_visit.append(absolute_link)

        return "\n".join(all_text)

    @staticmethod
    def extract_text_from_html(html_content):
        html_content = re.sub(r'<(script|style).*?</\1>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        return re.sub(r'\s+', ' ', html_content).strip()

class UIHelper:
    @staticmethod
    def render_latex(latex_str):
        return latex2mathml.converter.convert(latex_str)

    @staticmethod
    def display_message_content(content):
        parts = re.split(r'(\$\$.*?\$\$|\$.*?\$)', content, flags=re.DOTALL)
        for part in parts:
            if part.startswith('$') and part.endswith('$'):
                st.latex(part.strip('$'))
            else:
                st.write(part)

        if '```' in content:
            code_blocks = content.split('```')
            for i, block in enumerate(code_blocks):
                if i % 2 == 1:
                    language = block.split('\n')[0]
                    code = '\n'.join(block.split('\n')[1:])
                    st_ace(value=code, language=language, theme='monokai')

def main():
    st.title("🧠 DocuMind: Your Intelligent Document Assistant")

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem.load('rag_system.pkl') if os.path.exists('rag_system.pkl') else RAGSystem()

    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["txt", "docx", "pdf"])
        
        if uploaded_files:
            for file in uploaded_files:
                content = DocumentProcessor.process(file)
                st.session_state.rag_system.add_document(content)
            st.session_state.rag_system.save('rag_system.pkl')
            st.success(f"{len(uploaded_files)} document(s) uploaded and processed!")

        st.header("🌐 Web Crawler")
        url = st.text_input("Enter a URL to crawl:")
        max_pages = st.number_input("Maximum number of pages to crawl:", min_value=1, max_value=20, value=5)
        if st.button("Crawl and Add to RAG"):
            with st.spinner("Crawling website..."):
                crawled_content = WebCrawler.crawl(url, max_pages)
                summary = OpenAIWrapper.chat_completion([
                    {"role": "system", "content": "Summarize the following text concisely:"},
                    {"role": "user", "content": crawled_content[:2000]}
                ])
                st.session_state.rag_system.add_document(crawled_content)
                st.session_state.rag_system.save('rag_system.pkl')
                st.success("Website crawled, summarized, and added to RAG system!")
                st.write("Summary:", summary)

        if st.button("Clear All Documents"):
            st.session_state.rag_system.clear()
            if os.path.exists('rag_system.pkl'):
                os.remove('rag_system.pkl')
            st.success("All documents cleared!")
            st.experimental_rerun()

    st.header("💬 Chat with DocuMind")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.experimental_rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            UIHelper.display_message_content(message["content"])

    if prompt := st.chat_input("Ask DocuMind about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        relevant_chunks = st.session_state.rag_system.get_relevant_chunks(prompt)

        messages = [
            {"role": "system", "content": "You are DocuMind, an AI assistant. Provide helpful and informative responses based on the given context."},
            *st.session_state.messages[-5:],
            {"role": "user", "content": f"Context from documents:\n\n{' '.join(relevant_chunks)}\n\nUser question: {prompt}" if relevant_chunks else f"No relevant documents found. User question: {prompt}"}
        ]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.empty()
            UIHelper.display_message_content(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()