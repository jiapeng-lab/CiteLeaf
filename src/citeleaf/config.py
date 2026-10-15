"""One configuration boundary; credentials never enter application results."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_path: str = ""
    data_dir: Path = ROOT / "data"

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env")
        return cls(
            api_key=os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("LLM_MODEL", cls.model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_path=os.getenv("EMBEDDING_MODEL_PATH", ""),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve(),
        )
