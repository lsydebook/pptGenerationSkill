# -*- coding: utf-8 -*-
"""应用日志配置（固定默认值，不读 .env）。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging import LogRecord
from pathlib import Path

from src.config.env_loader import ROOT_DIR

_LOG_LEVEL = logging.INFO
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = ROOT_DIR / "logs"
_log_file: Path | None = None
_configured = False
_PROBE_ACCESS_PATHS = frozenset(
    {
        "/",
        "/json/version",
        "/json/list",
        "/favicon.ico",
    }
)


class DetailFormatter(logging.Formatter):
    """[时间] [进程[pid]] [级别] [线程] [文件:行号] 消息"""

    def format(self, record: LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        process_label = f"{_process_name()}[{record.process}]"
        location = f"{record.filename}:{record.lineno}"
        return (
            f"[{timestamp}] [{process_label}] [{record.levelname}] "
            f"[{record.threadName}] [{location}] {record.getMessage()}"
        )


class UvicornProbeAccessFilter(logging.Filter):
    """Drop Chrome DevTools / browser probes that hit this API port."""

    def filter(self, record: LogRecord) -> bool:
        path = _access_path(record)
        if path is None:
            return True
        return path.split("?", 1)[0] not in _PROBE_ACCESS_PATHS


def _access_path(record: LogRecord) -> str | None:
    args = record.args
    if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
        return args[2]
    message = record.getMessage()
    for marker in ('"GET ', '"HEAD '):
        start = message.find(marker)
        if start < 0:
            continue
        start += len(marker)
        end = message.find(" HTTP/", start)
        if end > start:
            return message[start:end]
    return None


def attach_uvicorn_probe_filter() -> None:
    """uvicorn 会在启动时重建 access logger，须在其完成配置后再挂过滤。"""
    access = logging.getLogger("uvicorn.access")
    if any(isinstance(item, UvicornProbeAccessFilter) for item in access.filters):
        return
    access.addFilter(UvicornProbeAccessFilter())


def _process_name() -> str:
    try:
        return Path(sys.argv[0]).stem.lower() or "app"
    except (IndexError, TypeError):
        return "app"


def _build_log_file_path() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d")
    return _LOG_DIR / f"{stamp}.log"


def setup_logging() -> None:
    """初始化根日志（幂等，应用启动时调用一次即可）。"""
    global _configured, _log_file
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)

    formatter = DetailFormatter(datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = _build_log_file_path()
    file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    _configured = True
    logging.getLogger(__name__).info("logging initialized file=%s", _log_file)


def get_logger(name: str) -> logging.Logger:
    """获取模块 logger；若尚未初始化则自动 setup。"""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)


def current_log_file() -> Path | None:
    """当前会话日志文件路径（setup 之后可用）。"""
    return _log_file
