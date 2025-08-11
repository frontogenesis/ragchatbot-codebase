import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location


config = Config()

# Validate critical configuration
if config.MAX_RESULTS <= 0:
    raise ValueError(f"MAX_RESULTS must be > 0, got {config.MAX_RESULTS}")

if not config.ANTHROPIC_API_KEY:
    print("WARNING: ANTHROPIC_API_KEY not set - queries will fail")

if config.MAX_RESULTS > 50:
    print(
        f"WARNING: MAX_RESULTS={config.MAX_RESULTS} may be too high and cause performance issues"
    )
