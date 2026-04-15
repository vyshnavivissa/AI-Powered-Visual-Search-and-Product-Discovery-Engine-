import os
from groq import Groq
import json
import re

# Set your API key properly (pick ONE of these approaches)

# APPROACH A — hardcode directly (quick for local dev, never commit to git)
client = Groq(api_key="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")

# APPROACH B — via environment variable (recommended for production)
# Step 1: In your terminal run:  set GROQ_API_KEY=gsk_vBRL...   (Windows)
# Step 2: Then use:
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_filters(query):
    prompt = f"""
Extract product filters from this query:

"{query}"

Return ONLY JSON with keys:
color, max_price, category, brand

Example:
{{
  "color": "red",
  "max_price": 2000,
  "category": "shirt",
  "brand": "nike"
}}

Rules:
- Return ONLY JSON, no explanation, no markdown
- If a field is not mentioned, set it to null
- Convert "under 2000" → max_price = 2000
- Detect brand names like Nike, Puma, Adidas, Roadster
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # valid Groq model (fast + free tier)
            # Other good options:
            # "llama3-70b-8192"       — more accurate, slightly slower
            # "mixtral-8x7b-32768"    — good at structured output
            # "gemma2-9b-it"          — lightweight alternative
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        text = response.choices[0].message.content.strip()
        print("LLM raw response:", text)   # helpful for debugging

        # Strip markdown code fences if model wraps in ```json ... ```
        text = re.sub(r"```(?:json)?", "", text).strip("`").strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

    except json.JSONDecodeError as e:
        print("JSON parse error:", e)
    except Exception as e:
        print("LLM error:", e)

    # Safe fallback
    return {
        "color": None,
        "max_price": None,
        "category": None,
        "brand": None
    }