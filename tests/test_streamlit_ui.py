"""Guards cho UI Streamlit.

Hai lớp kiểm tra:
- Chạy thật module `app/streamlit_app.py` với runtime Streamlit/httpx giả cho
  từng chế độ (auto / RAG thủ công / NLQ thủ công) để bắt lỗi kiểu NameError
  (`target_pipeline`, `route_info_str`) mà test tĩnh không thấy.
- Lỗi API hiển thị Trace ID và `detail`, không dán raw body.
"""

import ast
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

SOURCE = Path("app/streamlit_app.py").read_text(encoding="utf-8")

TRACE_ID = "d" * 32
ERROR_BODY = '{"detail": "provider exploded"}'

MODE_AUTO = "🤖 Tự động thông minh (Router)"
MODE_RAG = "📑 Hỏi đáp tài liệu (RAG)"
MODE_NLQ = "📊 Truy vấn CSDL bán hàng (NLQ)"


def _function(name: str):
    module = ast.parse(SOURCE)
    node = next(
        item
        for item in module.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {"json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<ui>", "exec"), namespace)
    return namespace[name]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _SessionState:
    def __init__(self):
        object.__setattr__(self, "_data", {})

    def __contains__(self, key):
        return key in self._data

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self._data[key] = value


class _FakeStreamlit(types.ModuleType):
    def __init__(self, mode: str):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.sidebar = self
        self._mode = mode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def radio(self, *args, **kwargs):
        return self._mode

    def selectbox(self, *args, **kwargs):
        return "fast (gemini-3.5-flash-lite)"

    def chat_input(self, *args, **kwargs):
        return "Câu hỏi thử"

    def feedback(self, *args, **kwargs):
        return None

    def file_uploader(self, *args, **kwargs):
        return None

    def chat_message(self, *args, **kwargs):
        return _Ctx()

    def status(self, *args, **kwargs):
        return _Ctx()

    def empty(self, *args, **kwargs):
        return _Ctx()

    def expander(self, *args, **kwargs):
        return _Ctx()

    def spinner(self, *args, **kwargs):
        return _Ctx()

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _FakeResponse:
    status_code = 500
    headers = {"X-Trace-Id": TRACE_ID}
    text = ERROR_BODY

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def json(self):
        raise ValueError("không có JSON khi lỗi")

    def read(self):
        return ERROR_BODY.encode()

    def iter_lines(self):
        return []


class _FakeClient(_Ctx):
    def __init__(self, *args, **kwargs):
        pass

    def stream(self, *args, **kwargs):
        return _FakeResponse()


def _run_ui(monkeypatch, mode: str):
    """Import và chạy toàn bộ streamlit_app với backend trả lỗi 500."""
    fake_st = _FakeStreamlit(mode)

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.get = lambda *args, **kwargs: _FakeResponse()
    fake_httpx.post = lambda *args, **kwargs: _FakeResponse()
    fake_httpx.Client = _FakeClient

    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)
    monkeypatch.delitem(sys.modules, "app.streamlit_app", raising=False)

    module = importlib.import_module("app.streamlit_app")
    messages = module.st.session_state.messages
    monkeypatch.delitem(sys.modules, "app.streamlit_app", raising=False)
    return messages


@pytest.mark.parametrize("mode", [MODE_AUTO, MODE_RAG, MODE_NLQ])
def test_each_ui_mode_renders_error_without_name_error(monkeypatch, mode):
    """Chọn từng chế độ trên UI không được ném NameError; lỗi phải có Trace ID."""
    messages = _run_ui(monkeypatch, mode)
    assistant = [msg for msg in messages if msg.get("role") == "assistant"]
    assert assistant, f"Chế độ {mode} không tạo thông báo lỗi nào"
    content = assistant[-1]["content"]
    assert "HTTP 500" in content
    assert "provider exploded" in content
    assert TRACE_ID in content
    assert ERROR_BODY not in content  # không dán raw body


def test_error_helper_shows_trace_id_and_detail():
    format_api_error = _function("format_api_error")
    message = format_api_error(
        504,
        {"X-Trace-Id": "0123456789abcdef0123456789abcdef"},
        '{"detail": "NLQ request timed out."}',
        "Lỗi khi truy vấn CSDL",
    )
    assert "HTTP 504" in message
    assert "NLQ request timed out." in message
    assert "0123456789abcdef0123456789abcdef" in message

    validation = format_api_error(
        422,
        {},
        '{"detail": [{"msg": "field required"}]}',
        "Dữ liệu không hợp lệ",
    )
    assert "field required" in validation
    assert "không có" in validation


def test_ui_error_paths_all_use_the_helper():
    """Mọi luồng chính đều đi qua helper hiển thị Trace ID thay vì raw body."""
    assert "target_pipeline" not in SOURCE
    assert "Lỗi máy chủ khi xử lý câu hỏi" in SOURCE
    assert "Lỗi máy chủ khi tra cứu tài liệu" in SOURCE
    assert "Lỗi khi truy vấn CSDL" in SOURCE
    assert "Lỗi xác thực dữ liệu feedback" in SOURCE
    assert SOURCE.count("format_api_error(") >= 5  # 1 định nghĩa + 4 chỗ gọi
