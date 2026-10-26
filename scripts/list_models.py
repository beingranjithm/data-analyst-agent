"""Lists the Gemini models your API key can use and checks that the model names in
agents/llm_factory.py exist. It is a LIST request: it does not generate text.
Run from the project root:  python scripts/list_models.py
Your key is read from hello_agent/.env and never printed."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / "hello_agent" / ".env")
key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not key:
    sys.exit("No API key found in hello_agent/.env (expected GOOGLE_API_KEY=...). Nothing was sent.")

from google import genai

from agents.llm_factory import ROLE_MODELS

client = genai.Client(api_key=key)
found = {}
for m in client.models.list():
    name = (getattr(m, "name", "") or "").replace("models/", "")
    actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
    found[name] = list(actions)

print(f"{len(found)} models visible to this key.\n")
print("Text models that can generate content (flash / lite / gemma):")
for name in sorted(found):
    if any(w in name for w in ("flash", "lite", "gemma")) and "generateContent" in found[name]:
        print("  ", name)

print("\nModels used by agents/llm_factory.py:")
missing = False
for role, names in ROLE_MODELS.items():
    for n in names:
        ok = n in found and "generateContent" in found[n]
        missing |= not ok
        print(f"   {role:8s} {n:28s} {'OK' if ok else 'NOT FOUND (or cannot generate)'}")
print("\nAll configured models exist." if not missing else "\nFix the names in ROLE_MODELS (agents/llm_factory.py) using the list above.")
