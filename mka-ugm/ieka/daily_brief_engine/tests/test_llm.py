import os
import sys
from pathlib import Path


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    # Prefer OpenRouter if configured
    model = os.getenv("MODEL", "openrouter/gpt-4o")
    try:
        from crewai import LLM
    except Exception as e:
        print("ERROR: crewai not importable:", e, file=sys.stderr)
        sys.exit(2)

    try:
        llm = LLM(model=model)
        resp = llm.call("Say hello from automated test")
        text = getattr(resp, "text", getattr(resp, "raw", resp))
        print("LLM RESPONSE:\n", text)
    except Exception as e:
        print("LLM call failed:", e, file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
