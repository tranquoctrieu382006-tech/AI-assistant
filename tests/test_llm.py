"""
Unit tests cho app.core.llm.
Kiểm tra logic phân tách provider, cấu hình model, cache, kwargs, timeout và error handling.
"""

from unittest.mock import MagicMock, patch
import pytest
from app.core.llm import (
    _parse_provider_and_model,
    _resolve_model_name,
    get_llm,
    get_rag_llm,
    clear_llm_cache,
    verify_model,
    check_ollama_status_async,
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

    def test_direct_model_name_resolves_for_configured_workflows(self):
        custom = "openai/custom-model-v1"
        assert _resolve_model_name(custom) == custom


def test_rag_llm_sets_explicit_context_for_ollama(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "rag_model", "local")
    monkeypatch.setattr(settings, "rag_num_ctx", 8192)
    with patch("app.core.llm.get_llm", return_value=object()) as builder:
        get_rag_llm()
    builder.assert_called_once_with("local", temperature=0.0, num_ctx=8192)


def test_rag_llm_does_not_pass_ollama_context_to_cloud_model(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "rag_model", "strong")
    with patch("app.core.llm.get_llm", return_value=object()) as builder:
        get_rag_llm()
    builder.assert_called_once_with("strong", temperature=0.0)


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

    def test_different_kwargs_different_instance(self):
        """Kiểm tra kwargs (ví dụ num_predict) được tính vào cache key chính xác."""
        llm1 = get_llm("local", num_predict=100, use_cache=True)
        llm2 = get_llm("local", num_predict=200, use_cache=True)
        assert llm1 is not llm2

    def test_invalid_provider_raises_llm_init_error(self):
        """Kiểm tra báo lỗi LLMInitError có gợi ý hữu ích khi provider không hợp lệ."""
        with pytest.raises(LLMInitError) as exc_info:
            get_llm("nonexistent_provider/invalid-model-name-xyz", use_cache=False)
        assert "Không khởi tạo được model" in str(exc_info.value)

    def test_verify_model_catches_runtime_exception(self):
        """Kiểm tra verify_model bắt mọi Exception khi invoke gặp sự cố (Ollama sập, 401...)."""
        with patch("app.core.llm.get_llm") as mock_get_llm:
            mock_instance = MagicMock()
            mock_instance.invoke.side_effect = RuntimeError("Mất kết nối máy chủ Ollama")
            mock_get_llm.return_value = mock_instance

            res = verify_model("local")
            assert res["status"] == "error"
            assert "Mất kết nối máy chủ Ollama" in res["error"]


class TestAsyncOllamaCheck:
    @pytest.mark.asyncio
    async def test_check_ollama_status_async_offline(self):
        """Kiểm tra check_ollama_status_async xử lý mượt mà khi port không tồn tại."""
        from app.core.config import settings
        with patch.object(settings, "ollama_base_url", "http://127.0.0.1:59999"):
            res = await check_ollama_status_async()
            assert res["status"] == "offline"
