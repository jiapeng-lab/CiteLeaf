"""Offline regression tests exercise storage and evidence boundaries, not API accuracy."""

import hashlib
import io
from dataclasses import replace

import httpx
import pytest
from pypdf import PdfWriter

from citeleaf.agent import DocumentAgent, bounded_history
from citeleaf.config import Settings
from citeleaf.documents import Chunk, parse_file, split_text
from citeleaf.generation import ChatModel, CitationError, ModelError, Source, validate
from citeleaf.knowledge import Knowledge


class TinyEncoder:
    """Test double only; production always uses a semantic BGE model."""

    def encode(self, texts, *, query=False):
        return [[1 + text.count("年假"), 1 + text.count("报销"), 1.0] for text in texts]


class ScriptedModel:
    def __init__(self, responses):
        self.responses, self.payloads = iter(responses), []

    def json(self, system, payload):
        self.payloads.append(payload)
        value = next(self.responses)
        return value(payload) if callable(value) else value


def supported(payload):
    source = payload["evidence"][0]
    return {
        "supported": True,
        "claims": [
            {"text": "测试结论", "evidence": [{"id": source["id"], "quote": source["text"][:20]}]}
        ],
    }


@pytest.fixture
def kb(tmp_path):
    return Knowledge(tmp_path / "index", TinyEncoder(), "test-double")


def test_chunks_preserve_entire_text_and_overlap():
    text = "".join(chr(0x4E00 + i) for i in range(1000))
    pieces = split_text(text, 100, 20)
    recovered = pieces[0] + "".join(p[20:] for p in pieces[1:])
    assert recovered == text
    assert max(map(len, pieces)) <= 100
    assert pieces[-1].endswith(text[-30:])


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_utf8_bom_and_stable_content_id():
    content = "\ufeff员工年假共12个工作日。".encode()
    first, second = parse_file(content, "a.txt"), parse_file(content, "b.txt")
    assert first[0].doc_id == second[0].doc_id == hashlib.sha256(content).hexdigest()
    assert first[0].page == 0 and not first[0].text.startswith("\ufeff")


@pytest.mark.parametrize(
    "content, filename",
    [(b"", "x.txt"), (b"\xff", "x.txt"), (b"123", "x.docx"), (b"broken", "x.pdf")],
)
def test_bad_upload_is_rejected(content, filename):
    with pytest.raises(ValueError):
        parse_file(content, filename)


def test_empty_pdf_requires_ocr():
    writer, output = PdfWriter(), io.BytesIO()
    writer.add_blank_page(width=300, height=300)
    writer.write(output)
    with pytest.raises(ValueError, match="OCR"):
        parse_file(output.getvalue(), "scanned.pdf")


def test_pdf_page_metadata():
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in ("Page one annual leave", "Page two reimbursement"):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    chunks = parse_file(output.getvalue(), "two.pdf")
    assert [c.page for c in chunks] == [1, 2]
    assert "Page two" in chunks[1].text


def test_storage_dedupe_scope_delete_and_reopen(kb, tmp_path):
    first, added = kb.ingest("年假为十二个工作日。".encode(), "a.txt")
    second, _ = kb.ingest("报销需要提交电子发票。".encode(), "b.txt")
    assert added
    assert not kb.ingest("年假为十二个工作日。".encode(), "renamed.txt")[1]
    assert len(kb.documents()) == 2
    assert all(c.doc_id == second for c in kb.search("年假", [second]))
    assert kb.search("年假", []) == []
    assert kb.read([]) == []
    reopened = Knowledge(tmp_path / "index", TinyEncoder(), "test-double")
    assert len(reopened.read([first])) == 1
    kb.delete(first)
    assert not kb.read([first]) and kb.read([second])


@pytest.fixture
def source():
    return Source("S1", Chunk("c", "d", "a.txt", 1, 0, "员工每年有12个工作日年假。"))


@pytest.mark.parametrize(
    "ref",
    [
        {"id": "S9", "quote": "员工每年有12个工作日年假。"},
        {"id": "S1", "quote": "这是模型编造的原文摘录"},
        {"id": "S1", "quote": "员工"},
        {"id": [], "quote": "123456"},
    ],
)
def test_fabricated_citations_fail_closed(source, ref):
    with pytest.raises(CitationError):
        validate({"supported": True, "claims": [{"text": "年假12天", "evidence": [ref]}]}, [source])


def test_supported_claim_requires_quotes(source):
    assert validate({"supported": False}, [source]) == []
    with pytest.raises(CitationError):
        validate({"supported": True, "claims": [{"text": "年假12天", "evidence": []}]}, [source])
    with pytest.raises(CitationError):
        validate({"supported": "true"}, [source])


def test_followup_retrieves_rewritten_query_and_bounds_history(kb):
    ident, _ = kb.ingest("年假需要提前五个工作日申请。".encode(), "a.txt")
    model = ScriptedModel([{"question": "年假提前多久申请？"}, supported])
    agent = DocumentAgent(kb, model)
    result = agent.run("提前多久？", [ident], [{"role": "user", "content": "年假有几天？"}])
    assert result.status == "ok"
    assert model.payloads[1]["task"] == "年假提前多久申请？"
    assert len(bounded_history([{"role": "user", "content": "x" * 2000}] * 20)) == 6
    assert len(bounded_history([{"role": "user", "content": "x" * 2000}])[0]["content"]) == 800


def test_unsupported_and_invalid_response(kb):
    ident, _ = kb.ingest("年假需要提前五个工作日申请。".encode(), "a.txt")
    agent = DocumentAgent(kb, ScriptedModel([{"supported": False}, {"supported": "bad"}]))
    assert agent.run("育儿假？", [ident]).status == "unsupported"
    assert agent.run("年假？", [ident]).status == "invalid"
    assert agent.run("年假？", []).status == "empty"


def test_summary_reads_tail_and_cache_invalidates_after_delete(kb):
    content = "前文制度说明。\n" * 650 + "最后规定：设备遗失应在两小时内报告。"
    ident, _ = kb.ingest(content.encode(), "long.txt")
    model = ScriptedModel([supported] * 30)
    agent = DocumentAgent(kb, model)
    first = agent.run("总结全文", [ident])
    assert first.status == "ok"
    map_payloads = [p for p in model.payloads if "draft_claims" not in p]
    assert any("最后规定" in e["text"] for p in map_payloads for e in p["evidence"])
    calls = len(model.payloads)
    assert agent.run("总结全文", [ident]).cached
    assert len(model.payloads) == calls
    kb.delete(ident)
    assert agent.run("总结全文", [ident]).status == "empty"


def test_model_errors_redact_credentials(monkeypatch):
    settings = Settings(api_key="TEST-SECRET")
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Response(401, text="TEST-SECRET"))
    with pytest.raises(ModelError) as caught:
        ChatModel(settings).json("system", {})
    assert "TEST-SECRET" not in str(caught.value)
    assert "TEST-SECRET" not in repr(settings)
    with pytest.raises(ModelError, match="设置"):
        ChatModel(replace(settings, api_key="")).json("system", {})


def test_streamlit_import_chat_and_scope_reset(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from citeleaf.config import ROOT

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "ui"))
    monkeypatch.setattr("citeleaf.knowledge.BGEEncoder", lambda path: TinyEncoder())
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
    assert not app.exception
    next(b for b in app.button if b.label == "导入演示资料").click().run(timeout=15)
    assert not app.exception and len(app.multiselect[0].value) == 2
    app.session_state["agent"].model = ScriptedModel([supported])
    app.chat_input[0].set_value("年假有多少天？").run(timeout=15)
    assert not app.exception
    assert len(app.session_state["messages"]) == 2
    assert app.session_state["messages"][-1]["result"]["status"] == "ok"
    app.multiselect[0].set_value([]).run(timeout=15)
    assert not app.exception
    assert app.session_state["messages"] == []
    assert app.chat_input[0].disabled


def test_provider_payload_and_truncated_output(monkeypatch):
    requests = []

    def respond(*args, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "length"}], "usage": {"total_tokens": 2400}}
        )

    monkeypatch.setattr(httpx, "post", respond)
    with pytest.raises(ModelError, match="长度上限"):
        ChatModel(Settings(api_key="test")).json("system", {})
    assert requests[-1]["thinking"] == {"type": "disabled"}
    with pytest.raises(ModelError):
        ChatModel(Settings(api_key="test", base_url="https://example.org/v1")).json("system", {})
    assert "thinking" not in requests[-1]
