"""
Unit tests cho app.core.llm.
Kiểm tra logic phân tách provider, cấu hình model, cache và error handling.
"""

import pytest
from app.core.llm import (
    _parse_provider_and_model,
    _resolve_model_name,
    get_llm,
    clear_llm_cache,
    LLMInitError,
    MODEL_REGISTRY,
)


class TestProviderParsing:
    def test_parse_with_slash(self):
        provider, model = _parse_provider_and_model("ollama/gemma3:12b")
        assert provider == "ollama"
        assert model == "gemma3:12b"

    def test_parse_openai_prefix(self):
        provider, model = _parse_provider_and_model("openai/gpt-4o-mini")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_parse_no_prefix(self):
        provider, model = _parse_provider_and_model("gpt-4o")
        assert provider is None
        assert model == "gpt-4o"

    def test_whitespace_handling(self):
        provider, model = _parse_provider_and_model("  ollama / qwen2.5:7b  ")
        assert provider == "ollama"
        assert model == "qwen2.5:7b"


class TestModelResolution:
    def test_resolve_default(self):
        from app.core.config import settings
        resolved = _resolve_model_name("default")
        assert resolved == settings.default_model

    def test_resolve_role(self):
        resolved = _resolve_model_name("local")
        assert resolved == MODEL_REGISTRY["local"]

    def test_resolve_direct_string(self):
        custom = "openai/custom-model-v1"
        resolved = _resolve_model_name(custom)
        assert resolved == custom


class TestLLMCacheAndError:
    def setup_method(self):
        clear_llm_cache()

    def test_instance_caching(self):
        """Kiểm tra get_llm tái sử dụng cùng instance khi cùng role/config."""
        llm1 = get_llm("local", use_cache=True)
        llm2 = get_llm("local", use_cache=True)
        assert llm1 is llm2

    def test_different_temperature_different_instance(self):
        llm1 = get_llm("local", temperature=0.0, use_cache=True)
        llm2 = get_llm("local", temperature=0.7, use_cache=True)
        assert llm1 is not llm2

    def test_invalid_provider_raises_llm_init_error(self):
        """Kiểm tra báo lỗi LLMInitError có gợi ý hữu ích khi provider không hợp lệ."""
        with pytest.raises(LLMInitError) as exc_info:
            get_llm("nonexistent_provider/invalid-model-name-xyz", use_cache=False)
        assert "Không khởi tạo được model" in str(exc_info.value)
