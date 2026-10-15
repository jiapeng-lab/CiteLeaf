"""Streamlit presentation only; all document reasoning lives in the package."""

import json
from pathlib import Path

import streamlit as st

from citeleaf.agent import DocumentAgent
from citeleaf.config import ROOT, Settings
from citeleaf.generation import ChatModel, ModelError
from citeleaf.knowledge import BGEEncoder, Knowledge

st.set_page_config(page_title="知页 CiteLeaf · 文档有据，回答有源", page_icon="🌿", layout="wide")
st.markdown(
    """<style>
:root{--green:#245b47;--paper:#f7f8f3}
.stApp{background:var(--paper);font-family:'Noto Sans SC',sans-serif;color:#253b32}
[data-testid="stSidebar"]{background:#edf1e8;border-right:1px solid #dbe4d6}
.block-container{max-width:1100px;padding-top:2.5rem;padding-bottom:2rem}
h1,h2,h3{letter-spacing:-.035em;color:#234d3c}
.eyebrow{font-size:11px;letter-spacing:3px;color:#738373;font-weight:700;margin-bottom:12px}
.hero{font-size:42px;font-weight:600;line-height:1.35;margin-bottom:14px}
.muted{color:#718073;font-size:14px;line-height:1.9}
.rule{border-top:1px solid #dce3d7;margin:24px 0}
.feature{border:1px solid #dce3d7;background:#fffdf8;padding:22px;border-radius:12px;min-height:125px}
.feature strong{display:block;color:#315542;margin-bottom:8px;font-size:16px}
.feature span{font-size:13px;line-height:1.8;color:#7a8578}
.stats{display:flex;gap:28px;margin:12px 0 24px}
.stats>div{flex:1;color:#718073;font-size:12px}
.stats strong{display:block;font-size:25px;font-weight:500;color:#285e49;margin-top:8px}
@media(max-width:640px){.hero{font-size:29px}.stats{gap:10px}.block-container{padding-top:2rem}}
[data-testid="stChatMessage"]{border:1px solid #e0e6da;background:#ffffffb8;border-radius:12px}
.stButton>button{border-radius:8px}
.stButton>button[kind="primary"]{background:#285e49;border-color:#285e49}
[data-testid="stMetricValue"]{font-size:25px;color:#285e49}
footer{visibility:hidden}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在准备本地向量模型，首次运行可能需要下载…")
def resources(data_dir, model_id, model_path):
    encoder = BGEEncoder(model_path or model_id)
    return Knowledge(Path(data_dir) / "chroma", encoder, model_id)


def show_result(result: dict):
    st.markdown(result["answer"])
    if result.get("cached"):
        st.caption("本次使用已校验的会话摘要缓存")
    sources = result.get("sources", [])
    if sources:
        with st.expander(f"核对原文 · {len(sources)} 个引用"):
            for source in sources:
                chunk = source["chunk"]
                location = (
                    f"第 {chunk['page']} 页" if chunk["page"] else f"片段 {chunk['index'] + 1}"
                )
                st.markdown(f"**[{source['id']}] {chunk['filename']} · {location}**")
                st.text(chunk["text"])
                quotes = list(
                    dict.fromkeys(
                        ref["quote"]
                        for claim in result.get("claims", [])
                        for ref in claim["evidence"]
                        if ref["id"] == source["id"]
                    )
                )
                for quote in quotes:
                    st.caption("已校验摘录：" + quote)
    with st.expander("查看执行步骤"):
        for i, step in enumerate(result.get("steps", []), 1):
            st.write(f"{i}. {step}")


settings = Settings.load()
try:
    kb = resources(str(settings.data_dir), settings.embedding_model, settings.embedding_path)
except Exception:
    st.error("本地向量模型初始化失败。请检查网络，或在 .env 设置 EMBEDDING_MODEL_PATH。")
    st.stop()
if "agent" not in st.session_state:
    st.session_state.agent = DocumentAgent(kb, ChatModel(settings))
    st.session_state.messages = []
    st.session_state.scope = []

with st.sidebar:
    st.markdown("## 知页 / CiteLeaf")
    st.caption("让每个答案，都有出处。")
    st.divider()
    st.markdown("#### 文档工作区")
    uploads = st.file_uploader(
        "上传你的资料",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="单文件不超过 15 MB；支持文字型 PDF、UTF-8 TXT",
    )
    if st.button("建立索引", use_container_width=True, disabled=not uploads):
        for upload in uploads:
            try:
                with st.spinner(f"正在处理 {upload.name}…"):
                    _, added = kb.ingest(upload.getvalue(), upload.name)
                st.success(f"{'已导入' if added else '已存在，跳过'}：{upload.name}")
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error(f"{upload.name} 索引失败，请检查模型和存储权限后重试")
    if st.button("导入演示资料", use_container_width=True):
        with st.spinner("正在为演示资料建立索引…"):
            try:
                for path in sorted((ROOT / "examples").glob("*.txt")):
                    kb.ingest(path.read_bytes(), path.name)
                st.success("两份虚构演示资料已就绪")
            except Exception:
                st.error("导入失败，请检查本地向量模型和存储权限")
    docs = kb.documents()
    names = {d["doc_id"]: d["filename"] for d in docs}
    ids = list(names)
    selected = st.multiselect(
        "本次回答使用的文档",
        ids,
        default=ids,
        format_func=lambda value: names[value],
        key=f"scope_{','.join(ids)}",
    )
    if selected != st.session_state.scope:
        st.session_state.messages = []
        st.session_state.scope = selected
        st.session_state.agent.cache.clear()
    st.caption("切换文档范围会开始新对话；清空选择时不检索。")
    with st.expander("管理文档"):
        for doc in docs:
            st.caption(f"{doc['filename']} · {doc['chunks']} 个片段")
            if st.button("删除此文档", key=f"delete_{doc['doc_id']}"):
                kb.delete(doc["doc_id"])
                st.session_state.messages = []
                st.session_state.agent.cache.clear()
                st.rerun()
    st.divider()
    mode = st.radio("任务方式", ["自动判断", "文档问答", "文档摘要"], horizontal=False)
    if st.button("开始新对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    with st.expander("运行配置"):
        st.caption("生成模型：" + settings.model)
        st.caption("向量模型：" + settings.embedding_model)
        st.caption("API 凭据：" + ("已配置" if settings.api_key else "未配置"))
        st.caption("文档向量保存在本地；生成时，相关原文会发送给配置的模型服务。")

st.markdown('<div class="eyebrow">CITELEAF / DOCUMENT INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">把资料，变成有据可查的答案。</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">阅读 · 提问 · 核对。专注你的文档，让结论回到原文。</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="stats"><div>工作区文档<strong>{len(docs)}</strong></div>'
    f"<div>当前选中<strong>{len(selected)}</strong></div>"
    f"<div>已索引片段<strong>{sum(d['chunks'] for d in docs)}</strong></div></div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.write("")
    for column, title, description in zip(
        st.columns(3),
        ["01 / 带着问题读", "02 / 从结论回到原文", "03 / 长文也能看全"],
        [
            "上传资料后直接提问，也可以接着上一轮继续追问。",
            "逐条结论附带引用，展开即可核对文件、页码和原文。",
            "全文顺序读取、分批摘要。也可以只总结你关心的部分。",
        ],
        strict=True,
    ):
        column.markdown(
            f'<div class="feature"><strong>{title}</strong><span>{description}</span></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.caption("第一次使用？在左侧导入演示资料，再试试：")
    for column, sample in zip(
        st.columns(3), ["年假有多少天？", "总结全部文档", "公司有育儿假吗？"], strict=True
    ):
        if column.button(sample, use_container_width=True, disabled=not selected):
            st.session_state.pending = sample

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_result(message["result"])

question = st.chat_input("向文档提问，或输入“总结报销流程”…", disabled=not selected)
question = question or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在查阅文档并核对引用…"):
                result = st.session_state.agent.run(
                    question,
                    selected,
                    history,
                    mode={"自动判断": "auto", "文档问答": "qa", "文档摘要": "summary"}[mode],
                ).export()
            show_result(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "result": result}
            )
        except ModelError as exc:
            st.error(str(exc))
            st.session_state.messages.pop()
        except Exception:
            st.error("处理失败，请检查本地模型或文档存储后重试。")
            st.session_state.messages.pop()

if st.session_state.messages:
    st.download_button(
        "导出本次对话与引用 · JSON",
        json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
        "citeleaf-conversation.json",
        "application/json",
    )
st.caption("知页 CiteLeaf · 原文摘录校验通过 ≠ 语义必然正确，关键结论请核对来源。")
