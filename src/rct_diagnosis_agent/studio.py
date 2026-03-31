from __future__ import annotations

from dotenv import load_dotenv

from .agent import build_default_agent

load_dotenv()
graph = build_default_agent().graph
