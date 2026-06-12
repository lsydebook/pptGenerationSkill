# -*- coding: utf-8 -*-
"""环境变量加载与读取工具。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# 项目根目录下的环境变量文件
ENV_PATH = ROOT_DIR / ".env"


def project_root() -> Path:
    return ROOT_DIR

load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes"}
