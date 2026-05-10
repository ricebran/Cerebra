"""Configuration management for the application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Qdrant Vector DB Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# PostgreSQL Configuration
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/app")

# Feature Flags
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS Configuration
# In production, set this to your frontend domain(s)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if ALLOWED_ORIGINS == ["*"]:
    # Keep wildcard for development, but log warning
    print("WARNING: CORS allows all origins (*). Configure ALLOWED_ORIGINS for production.")
