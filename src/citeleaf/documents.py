"""Pure PDF/TXT parsing and page-preserving chunking."""

import hashlib
import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    filename: str
    page: int  # 0 means TXT, PDF starts at 1
    index: int
    text: str

    def metadata(self):
        return {k: v for k, v in asdict(self).items() if k not in {"id", "text"}}


def split_text(text: str, size: int = 420, overlap: int = 60) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("分块参数需要满足 0 ≤ overlap < size")
    text = text.replace("\x00", "").replace("\r\n", "\n").strip()
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"\n|[。！？；.!?]", text[start:end]))
            if boundaries and boundaries[-1].end() > size // 2:
                end = start + boundaries[-1].end()
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def parse_file(content: bytes, filename: str) -> list[Chunk]:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("文件为空或超过 15 MB 上限")
    filename = Path(filename).name
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            pages = [(0, content.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("TXT 请保存为 UTF-8 编码后重试") from exc
    elif suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("请上传已解密的 PDF")
            if len(reader.pages) > 300:
                raise ValueError("演示版最多支持 300 页 PDF")
            pages = [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF 解析失败，请确认文件没有损坏") from exc
    else:
        raise ValueError("仅支持 PDF 或 TXT 文件")
    doc_id = hashlib.sha256(content).hexdigest()
    chunks = []
    for page, text in pages:
        for piece in split_text(text):
            index = len(chunks)
            chunks.append(Chunk(f"{doc_id}:{index}", doc_id, filename, page, index, piece))
    if not chunks:
        raise ValueError("未提取到文字；扫描 PDF 请先使用 OCR 转为可搜索 PDF")
    return chunks
