from fastapi import FastAPI
from pydantic import BaseModel
from ssd_llm_cpp_integration import SSDEnhancedLLM, SSDLLMConfig

app = FastAPI(title="SSD Chat API")
engine = SSDEnhancedLLM(SSDLLMConfig())

class Inp(BaseModel):
    text: str

@app.post("/generate")
def generate(inp: Inp):
    if inp.text.startswith("/say "):
        reply = inp.text[5:]
        meta = {"mode_used": "verbatim_bypass"}
    else:
        out = engine.generate_response(inp.text)
        reply, meta = out["response"], out["ssd_metadata"]
    return {"text": reply, "meta": meta}
