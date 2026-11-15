"""One configuration boundary; credentials never enter application results."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool = False) -> bool:
    """把环境变量解析为布尔值；未设置时用默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """把环境变量解析为整数；缺失或非法时回退到默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """集中保存模型、凭据和本地数据目录配置。"""
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_path: str = ""
    data_dir: Path = ROOT / "data"
    # 唯一的路由开关：开 = BGE + BM25 混合检索，关 = 单独 BGE。两条路线都会重排。
    hybrid_search: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 20
    top_k: int = 6

    @classmethod
    def load(cls):
        """加载项目 .env 和环境变量，生成应用配置。"""
        # 统一从环境变量读取配置；密钥字段不出现在 Settings 的 repr 中。
        load_dotenv(ROOT / ".env")
        return cls(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("LLM_MODEL", cls.model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_path=os.getenv("EMBEDDING_MODEL_PATH", ""),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve(),
            hybrid_search=_env_bool("HYBRID_SEARCH"),
            rerank_model=os.getenv("RERANK_MODEL", cls.rerank_model),
            rerank_top_n=_env_int("RERANK_TOP_N", cls.rerank_top_n),
            top_k=_env_int("TOP_K", cls.top_k),
        )
