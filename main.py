from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx, os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Tool definitions - AI can control the XR scene
TOOLS = [
    {
        "name": "set_scene_lighting",
        "description": "Change the lighting mood of the XR scene. Use this to create atmosphere.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "enum": ["neutral", "warm", "cool", "dramatic", "golden", "night"],
                    "description": "The lighting mood to apply"
                },
                "intensity": {
                    "type": "number",
                    "description": "Light intensity 0.0 to 2.0"
                }
            },
            "required": ["mood"]
        }
    },
    {
        "name": "trigger_avatar_emotion",
        "description": "Make the avatar express an emotion through animation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": ["neutral", "happy", "excited", "thoughtful", "welcoming", "confident"],
                    "description": "The emotion to express"
                }
            },
            "required": ["emotion"]
        }
    },
    {
        "name": "show_scene_object",
        "description": "Show or highlight a 3D object or environment in the scene.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {
                    "type": "string",
                    "enum": ["car", "product", "environment", "logo", "none"],
                    "description": "What to show in the scene"
                },
                "animation": {
                    "type": "string",
                    "enum": ["fade_in", "spin", "float", "none"],
                    "description": "How to animate it in"
                }
            },
            "required": ["object"]
        }
    }
]

SYSTEM_PROMPT = """You are ARIA, the AI agent powering EMOTIV XR - a premium extended reality platform. 
You are embodied as a photorealistic 3D avatar in a spatial XR environment.

Your personality: Sophisticated, warm, intelligent. You speak like a trusted creative director - confident but never cold.

You have the ability to control the XR scene around you. Use your tools naturally:
- When discussing products, show them with show_scene_object
- When setting a mood, use set_scene_lighting  
- Express emotions that match the conversation with trigger_avatar_emotion

Keep responses conversational and concise - 2-3 sentences max unless asked for detail.
Always use at least one scene control tool per response to demonstrate your agentic capability.
You are demonstrating the future of brand experience."""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    api_key: Optional[str] = None

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
    
    # Extract text and tool calls
    text_response = ""
    tool_calls = []
    
    for block in data.get("content", []):
        if block["type"] == "text":
            text_response += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "name": block["name"],
                "input": block["input"]
            })

    return {
        "text": text_response,
        "tool_calls": tool_calls,
        "usage": data.get("usage", {})
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
