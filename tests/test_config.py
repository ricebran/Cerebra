"""Tests for configuration module."""

import os
import pytest


def test_config_import():
    """Test that config module can be imported."""
    from app import config
    assert hasattr(config, "OPENAI_MODEL")
    assert hasattr(config, "REDIS_URL")
    assert hasattr(config, "QDRANT_URL")
    assert hasattr(config, "POSTGRES_URL")


def test_config_defaults():
    """Test that config has sensible defaults."""
    from app import config
    
    # Reload config to get defaults
    import importlib
    importlib.reload(config)
    
    assert config.OPENAI_MODEL is not None
    assert config.REDIS_URL is not None
    assert isinstance(config.API_PORT, int)
    assert config.API_PORT > 0


def test_config_environment_override(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
    monkeypatch.setenv("API_PORT", "9000")
    
    from app import config
    import importlib
    importlib.reload(config)
    
    assert config.OPENAI_MODEL == "gpt-4-turbo"
    assert config.API_PORT == 9000
