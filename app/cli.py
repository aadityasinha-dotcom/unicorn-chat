"""Interactive terminal client. Usage: python -m app.cli [--session ID]"""
from __future__ import annotations

import argparse
import sys

from app.bootstrap import build_engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the Indian unicorn startups dataset")
    parser.add_argument("--session", help="Resume / name a session id")
    parser.add_argument("--show-metrics", action="store_true", help="Print metrics on exit")
    args = parser.parse_args()

    engine = build_engine()
    session_id = args.session
    print(f"Loaded {len(engine.repo)} companies. Model: {engine.model.model_name}. Type 'exit' to quit.\n")
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                break
            if user.strip().lower() in {"exit", "quit", "q"}:
                break
            result = engine.chat(user, session_id)
            session_id = result.session_id
            tag = "" if result.kind == "answer" else f" [{result.kind}]"
            print(f"bot{tag}> {result.reply}\n      ({result.latency_ms:.0f} ms, tools={result.tool_calls or '-'})\n")
    except KeyboardInterrupt:
        pass
    finally:
        engine.tracer.flush()
        if args.show_metrics:
            import json

            print(json.dumps(engine.metrics.snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
