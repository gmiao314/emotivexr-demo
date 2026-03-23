from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TOOLS = [
    {
        "name": "set_scene_lighting",
        "description": "Change the lighting mood of the XR scene.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mood": {"type": "string", "enum": ["neutral","warm","cool","dramatic","golden","night"]},
                "intensity": {"type": "number"}
            },
            "required": ["mood"]
        }
    },
    {
        "name": "trigger_avatar_emotion",
        "description": "Make the avatar express an emotion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "emotion": {"type": "string", "enum": ["neutral","happy","excited","thoughtful","welcoming","confident"]}
            },
            "required": ["emotion"]
        }
    },
    {
        "name": "show_scene_object",
        "description": "Show a 3D object in the scene.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {"type": "string", "enum": ["car","product","environment","logo","none"]},
                "animation": {"type": "string", "enum": ["fade_in","spin","float","none"]}
            },
            "required": ["object"]
        }
    }
]

SYSTEM_PROMPT = "You are ARIA, the AI agent powering EMOTIV XR - a premium extended reality platform. You are embodied as a 3D avatar in a spatial XR environment. Personality: sophisticated, warm, intelligent - like a trusted creative director. Use your tools naturally every response. Keep responses to 2-3 sentences max. You represent the future of brand experience."

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    api_key: Optional[str] = None

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/index.html")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    key = req.api_key or ANTHROPIC_KEY
    if not key:
        raise HTTPException(400, "No Anthropic API key")

    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "tools": TOOLS,
                "messages": messages
            }
        )

    if not r.is_success:
        raise HTTPException(r.status_code, f"Claude error: {r.text}")

    data = r.json()
    text_response = ""
    tool_calls = []

    for block in data.get("content", []):
        if block["type"] == "text":
            text_response += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({"name": block["name"], "input": block["input"]})

    return {"text": text_response, "tool_calls": tool_calls}

app.mount("/static", StaticFiles(directory="static"), name="static")
