"""Offline regression tests exercise storage and evidence boundaries, not API accuracy."""

import hashlib
import io
from dataclasses import replace

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from openai import APIConnectionError, AuthenticationError
from pypdf import PdfWriter

from citeleaf.agent import DocumentAgent, bounded_history
from citeleaf.config import Settings
from citeleaf.documents import Chunk, parse_file, split_text
from citeleaf.generation import ChatModel, CitationError, ModelError, Source, validate
from citeleaf.knowledge import Knowledge


class TinyEncoder(Embeddings):
    """Test double only; production always uses a semantic BGE model."""

    def encode(self, texts, *, query=False):
        return [[1 + text.count("年假"), 1 + text.count("报销"), 1.0] for text in texts]

    def embed_documents(self, texts):
        return self.encode(texts)

    def embed_query(self, text):
        return self.encode([text], query=True)[0]


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


def test_model_errors_redact_credentials():
    settings = Settings(api_key="TEST-SECRET")

    def unauthorized(value, **kwargs):
        raise AuthenticationError(
            "TEST-SECRET",
            response=httpx.Response(401, request=httpx.Request("POST", "https://example.org")),
            body={"error": "TEST-SECRET"},
        )

    with pytest.raises(ModelError) as caught:
        ChatModel(settings, RunnableLambda(unauthorized)).json("system", {})
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


def test_provider_selection_and_truncated_output():
    from langchain_deepseek import ChatDeepSeek
    from langchain_openai import ChatOpenAI

    model = ChatModel(Settings(api_key="test"))
    provider = model._build_model()
    assert isinstance(provider, ChatDeepSeek)
    assert provider.extra_body == {"thinking": {"type": "disabled"}}
    other = ChatModel(Settings(api_key="test", base_url="https://example.org/v1"))._build_model()
    assert isinstance(other, ChatOpenAI) and not isinstance(other, ChatDeepSeek)
    assert other.extra_body == {}
    truncated = RunnableLambda(
        lambda value, **kwargs: AIMessage(
            content='{"supported":',
            response_metadata={"finish_reason": "length"},
            usage_metadata={"input_tokens": 100, "output_tokens": 2400, "total_tokens": 2500},
        )
    )
    model = ChatModel(Settings(api_key="test"), truncated)
    with pytest.raises(ModelError, match="长度上限"):
        model.json("system", {})
    assert model.calls == 1 and model.tokens == 2500


def test_langchain_prompt_parser_and_tracing_boundary():
    import json

    from langsmith import get_tracing_context

    def respond(prompt, **kwargs):
        messages = prompt.to_messages()
        assert messages[0].content == '规则包含JSON花括号：{"a":1}'
        assert json.loads(messages[1].content) == {"question": "年假有几天？"}
        assert kwargs["response_format"] == {"type": "json_object"}
        assert get_tracing_context()["enabled"] is False
        return AIMessage(content='{"supported":false,"claims":[]}')

    model = ChatModel(Settings(api_key="test"), RunnableLambda(respond))
    assert model.json('规则包含JSON花括号：{"a":1}', {"question": "年假有几天？"}) == {
        "supported": False,
        "claims": [],
    }


def test_partial_json_is_not_repaired_into_success():
    model = ChatModel(
        Settings(api_key="test"),
        RunnableLambda(lambda value, **kwargs: AIMessage(content='{"supported":false')),
    )
    with pytest.raises(ModelError, match="完整 JSON"):
        model.json("system", {})


def test_network_retry_is_bounded(monkeypatch):
    monkeypatch.setattr("citeleaf.generation.time.sleep", lambda delay: None)

    def disconnected(value, **kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "https://example.org"))

    model = ChatModel(Settings(api_key="test"), RunnableLambda(disconnected))
    with pytest.raises(ModelError, match="网络"):
        model.json("system", {})
    assert model.calls == 2


def test_existing_raw_chroma_index_is_readable_after_migration(tmp_path):
    import chromadb
    from langchain_chroma import Chroma

    directory = tmp_path / "legacy"
    client = chromadb.PersistentClient(path=str(directory))
    fingerprint = hashlib.sha256(b"test-double:420:60:v1").hexdigest()[:12]
    collection = client.get_or_create_collection(
        f"citeleaf_{fingerprint}", embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    chunk = parse_file("员工年假需要提前五个工作日申请。".encode(), "legacy.txt")[0]
    collection.add(
        ids=[chunk.id],
        documents=[chunk.text],
        metadatas=[chunk.metadata()],
        embeddings=TinyEncoder().embed_documents([chunk.text]),
    )
    kb = Knowledge(directory, TinyEncoder(), "test-double")
    assert isinstance(kb.vectorstore, Chroma)
    assert kb.search("年假申请", [chunk.doc_id]) == [chunk]
    assert not kb.ingest("员工年假需要提前五个工作日申请。".encode(), "legacy.txt")[1]
