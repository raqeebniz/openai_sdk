from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional
from datetime import datetime, UTC
from uuid import uuid4


# Initialize the FastAPI app
app = FastAPI(
    title= "DACA Chatbot API",
    description="A FastAPI-based API for a chatbot in DACA tutorial series",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],    # Allow frontend origin (e.g., React app)
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods
    allow_headers=["*"],   # Allow all headers
)

# Mock dictionary to store messages (Exercise 1 ka Step 1)
mock_message_db = {}

# Complex Pydentic models 
class Metadata(BaseModel):
    timestamp: datetime = Field(default_factory= lambda: datetime.now(UTC))
    session_id: str = Field(default_factory=lambda: str(uuid4()))


class Message(BaseModel):
    user_id: str
    text: str 
    metadata: Metadata
    tags: Optional[List[str]] = None   # Optional list of tags

    # Custom validator for text field
    @field_validator('text')
    def text_must_be_less_than_500_chars(cls, v):
        if len(v) > 500:
            raise ValueError('Text must not exceed 500 characters')
        return v


class Response(BaseModel):
    user_id: str
    reply: str
    metadata: Metadata


# Simulate a database dependecy
async def get_db():
    return {"connection": "Mock DB Connection"}


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the DACA Chatbot API! Access /docs for the API documentation."}

# Get endpoint with query parameters
@app.get("/users/{user_id}")
async def get_user(user_id: str, role: str | None = None):
    user_info = {"user_id": user_id, "role": role if role else "guest"}
    return user_info


# Post endpoint for chatting
@app.post("/chat/", response_model = Response)
async def chat(message: Message, db: dict = Depends(get_db)):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be empty")
    print(f"DB Connection: {db['connection']}")

    user_id = message.user_id
    if user_id not in mock_message_db:
        mock_message_db[user_id] = []
    mock_message_db[user_id].append(message)

    reply_text = f"Hello, {message.user_id}! You said: '{message.text}?'. How can I assist you today?"
    return Response(
        user_id = message.user_id,
        reply = reply_text,
        metadata=Metadata()   # Auto-generate timestamp and session_id
    )



# New endpoint to retrieve messages (Exercise 1 ka Step 3)
@app.get("/messages/{user_id}")
async def get_messages(user_id: str):
    if user_id not in mock_message_db:
        raise HTTPException(status_code=404, detail="User not found")
    return mock_message_db[user_id]