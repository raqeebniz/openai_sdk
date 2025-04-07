from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime
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



# Complex Pydentic models 
