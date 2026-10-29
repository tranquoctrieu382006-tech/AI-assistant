"""
Router / Intent classification module for AI Assistant.

Phân loại câu hỏi của người dùng thành một trong 3 nhánh:
- 'rag': Câu hỏi về văn bản tài liệu, chính sách bán hàng, đổi trả, bảo hành, vận chuyển, quy trình nội bộ.
- 'nlq': Câu hỏi về số liệu có cấu trúc, đơn hàng, doanh thu, khách hàng, sản phẩm, tồn kho, bảng biểu CSDL.
- 'out_of_scope': Câu hỏi ngoài phạm vi nghiệp vụ (thời tiết, chào hỏi, câu hỏi chung không liên quan đến dữ liệu/tài liệu).
"""

from enum import Enum
import json
import re
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import settings
from app.core.llm import get_llm


class RouteCategory(str, Enum):
    RAG = "rag"
    NLQ = "nlq"
    OUT_OF_SCOPE = "out_of_scope"


class RouteDecision(BaseModel):
    category: RouteCategory = Field(description="Phân loại câu hỏi: rag, nlq, hoặc out_of_scope")
    reasoning: str = Field(default="", description="Giải thích ngắn gọn lý do phân loại")


ROUTER_SYSTEM_PROMPT = """Bạn là chuyên gia phân loại câu hỏi (Router) cho hệ thống AI Doanh nghiệp.
Nhiệm vụ của bạn là đọc câu hỏi của người dùng và phân loại chính xác vào một trong ba nhóm sau:

1. 'rag':
- Hỏi về nội dung tài liệu, văn bản, thông tư, hợp đồng.
- Chính sách bán hàng, chính sách đổi trả hàng, quy trình bảo hành, quy chuẩn vận chuyển, lưu kho.
- Các câu hỏi dạng: "chính sách...", "quy định...", "hướng dẫn...", "điều kiện để...", "thời hạn đổi hàng...".

2. 'nlq':
- Hỏi về số liệu cụ thể trong cơ sở dữ liệu bán hàng.
- Đơn hàng, doanh thu, giá trị bán, số lượng sản phẩm, danh sách khách hàng, tồn kho kho hàng.
- Có các từ khóa định lượng hoặc tính toán: "bao nhiêu", "tổng", "top", "doanh thu tháng", "ai mua nhiều nhất", "sản phẩm nào bán chạy nhất".

3. 'out_of_scope':
- Câu chào hỏi xã giao (xin chào, hello, bạn là ai...).
- Câu hỏi ngoài phạm vi kinh doanh (thời tiết, thơ ca, viết code, toán học đại cương...).
- Câu hỏi rỗng hoặc vô nghĩa.

Hãy trả về JSON:
{"category": "rag" | "nlq" | "out_of_scope", "reasoning": "giải thích"}
"""


import unicodedata


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để đối sánh linh hoạt cả câu có dấu và không dấu."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D").lower()


def _rule_based_classify(question: str) -> RouteDecision:
    """Fallback rule-based heuristic classification khi không có LLM."""
    raw_q = (question or "").strip()
    q = raw_q.lower()
    if not q:
        return RouteDecision(category=RouteCategory.OUT_OF_SCOPE, reasoning="Câu hỏi rỗng")

    # Chuẩn hóa bỏ dấu câu và chuẩn hóa không dấu
    q_clean = re.sub(r"[^\w\s]", " ", q).strip()
    q_no_accents = strip_accents(q_clean)

    if len(q_no_accents) < 3:
        return RouteDecision(category=RouteCategory.OUT_OF_SCOPE, reasoning="Câu hỏi quá ngắn/vô nghĩa")

    # 1. Kiểm tra chào hỏi xã giao hoặc ngoài phạm vi (hỗ trợ cả có dấu và không dấu: "chao ban", "chào bạn",...)
    greeting_keywords = [
        "xin chao", "chao ban", "chao buoi", "chao", "hello", "hi", "ban la ai",
        "thoi tiet", "ke chuyen", "lam tho", "abc", "test", "alo", "tro ly",
        "cam on", "thanks", "thank you", "bye", "tam biet", "giup toi", "cho hoi"
    ]
    for gk in greeting_keywords:
        pattern = rf"\b{re.escape(gk)}\b"
        if q_no_accents == gk or q_no_accents.startswith(gk + " ") or (len(q_no_accents) < 25 and re.search(pattern, q_no_accents)):
            return RouteDecision(category=RouteCategory.OUT_OF_SCOPE, reasoning=f"Khớp mẫu chào hỏi/xã giao: {gk}")

    # 2. Kiểm tra từ khóa tài liệu/chính sách (RAG) (hỗ trợ cả có dấu và không dấu)
    rag_patterns = [
        r"chinh sach", r"quy dinh", r"bao hanh", r"doi tra", r"van chuyen",
        r"huong dan", r"thu tuc", r"dieu kien", r"hop dong", r"tai lieu",
        r"bao mat", r"thoi han", r"lam sao de", r"nhu the nao", r"quy trinh",
        r"chính sách", r"quy định", r"bảo hành", r"đổi trả", r"vận chuyển",
        r"hướng dẫn", r"thủ tục", r"điều kiện", r"hợp đồng", r"tài liệu"
    ]
    for pattern in rag_patterns:
        if re.search(pattern, q_clean) or re.search(pattern, q_no_accents):
            return RouteDecision(category=RouteCategory.RAG, reasoning=f"Khớp từ khóa tài liệu/chính sách: {pattern}")

    # 3. Từ khóa NLQ (truy vấn CSDL số liệu) (hỗ trợ cả có dấu và không dấu)
    nlq_patterns = [
        r"doanh thu", r"bao nhieu", r"tong so", r"top \d+", r"khach hang",
        r"san pham", r"don hang", r"ton kho", r"ban chay", r"dat nhat",
        r"chi phi", r"thang \d+", r"nam \d{4}", r"luong ban", r"trung binh",
        r"so luong", r"đơn hàng", r"khách hàng", r"sản phẩm", r"tồn kho",
        r"bao nhiêu", r"tổng", r"kho hang", r"nha cung cap", r"danh muc",
        r"kho hàng", r"nhà cung cấp", r"danh mục", r"gia ban", r"giá bán",
        r"doanh so", r"doanh số", r"nhap kho", r"xuat kho", r"nhập kho", r"xuất kho"
    ]
    for pattern in nlq_patterns:
        if re.search(pattern, q_clean) or re.search(pattern, q_no_accents):
            return RouteDecision(category=RouteCategory.NLQ, reasoning=f"Khớp từ khóa số liệu: {pattern}")

    # Nếu câu hỏi ngắn (< 15 ký tự) mà không khớp RAG hay NLQ -> xếp vào ngoài phạm vi
    if len(q_no_accents) < 15:
        return RouteDecision(category=RouteCategory.OUT_OF_SCOPE, reasoning="Câu hỏi ngắn ngoài phạm vi tài liệu/số liệu")

    return RouteDecision(category=RouteCategory.RAG, reasoning="Mặc định tra cứu tài liệu")


def route_question(question: str, *, llm: Any = None) -> RouteDecision:
    """
    Phân loại câu hỏi của người dùng.
    
    Hỗ trợ cả Rule-based Heuristic siêu tốc và LLM (thật hoặc FakeListChatModel).
    """
    clean_q = (question or "").strip()
    if not clean_q:
        return RouteDecision(category=RouteCategory.OUT_OF_SCOPE, reasoning="Câu hỏi rỗng")

    # 1. Ưu tiên phân loại siêu tốc qua rule-based cho chào hỏi và từ khóa rõ ràng
    rule_decision = _rule_based_classify(clean_q)
    if rule_decision.category == RouteCategory.OUT_OF_SCOPE:
        return rule_decision
    if "Khớp từ khóa" in rule_decision.reasoning and llm is None:
        return rule_decision

    # 2. Nếu truyền LLM cụ thể (ví dụ FakeListChatModel trong test)
    if llm is not None:
        try:
            msg = llm.invoke([
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=clean_q)
            ])
            text = getattr(msg, "content", str(msg)).strip()

            # Thử parse JSON
            try:
                json_match = re.search(r"\{.*?\}", text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    cat = data.get("category", "").lower()
                    if cat in ("rag", "nlq", "out_of_scope"):
                        return RouteDecision(
                            category=RouteCategory(cat),
                            reasoning=data.get("reasoning", "Phân loại từ JSON model")
                        )
            except Exception:
                pass

            # Fallback text matching trên output model
            low_text = text.lower()
            if "out_of_scope" in low_text or "ngoài phạm vi" in low_text:
                return RouteDecision(category=RouteCategory.OUT_OF_SCOPE, reasoning="Phân loại ngoài phạm vi")
            if "nlq" in low_text:
                return RouteDecision(category=RouteCategory.NLQ, reasoning="Phân loại NLQ từ model")
            if "rag" in low_text:
                return RouteDecision(category=RouteCategory.RAG, reasoning="Phân loại RAG từ model")

            return rule_decision
        except Exception:
            return rule_decision

    # 3. Thử dùng mô hình cấu hình 'fast' nếu không khớp rule-based
    try:
        active_llm = get_llm(role="fast")
        return route_question(clean_q, llm=active_llm)
    except Exception:
        return rule_decision
