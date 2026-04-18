# src/config.py
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Collections – navnene kan du bruke på tvers
TEXT_COLLECTION_NAME = "pdf_knowledge"
FIGURE_COLLECTION_NAME = "pdf_figures_clip"