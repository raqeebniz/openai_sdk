 
import pytest 
from fastapi.testclient import TestClient
from main import app

# client a test client
client = TestClient(app)

# Test the root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the DACA Chatbot API! Access /docs for the API documentation."
    }

# Test the /user/{user_id} endpoint    
def test_get_user():
    response = client.get("/users/alice?role=admin")
    assert response.status_code == 200
    assert response.json() == {"user_id": "alice", "role": "admin"}

    response = client.get("/users/bob")
    assert response.status_code == 200
    assert response.json() == {"user_id": "bob", "role": "guest"}


# Test the /chat/ endpoint (async test)
@pytest.mark.asyncio
async def test_chat():
    # Valic request
    request_data = {
        "user_id": "alice",
        "text": "Hello, how are you",
        "metadata": {
            "timestamp": "2025-04-06T12:00:00Z",
            "session_id": "123e4567-e89b-12d3-a456-426614174000"
        },
        "tags": ["Greeting"] 
    }
    response = client.post("/chat/", json=request_data)
    assert response.status_code == 200
    assert response.json()["user_id"] == "alice"
    assert response.json()["reply"] == "Hello, alice! You said: 'Hello, how are you?'. How can I assist you today?"
    assert "metadata" in response.json()

    # Invalid request (empty text)
    invalid_data = {
        "user_id": "bob",
        "text": "",
        "metadata": {
            "timestamp": "2025-04-06T12:00:00Z",
            "session_id": "123e4567-e89b-12d3-a456-426614174001"
        }
    }

    response= client.post("/chat/", json=invalid_data)
    assert response.status_code == 400
    assert response.json() == {"detail": "Message text cannot be empty"}




# New Code: Test for invalid timestamp format
@pytest.mark.asyncio
async def test_chat_invalid_timestamp():
    invalid_timestamp_data = {
        "user_id": "alice",
        "text": "Hello",
        "metadata": {
            "timestamp": "invalid-date",  # Invalid timestamp format
            "session_id": "123e4567-e89b-12d3-a456-426614174000"
        },
        "tags": ["Greeting"]
    }
    response = client.post("/chat/", json=invalid_timestamp_data)
    assert response.status_code == 422  # Expecting validation error
    assert "metadata.timestamp" in str(response.json())  # Check if error is related to timestamp

# New Code: Test for missing user_id
@pytest.mark.asyncio
async def test_chat_missing_user_id():
    missing_user_id_data = {
        "text": "Hello",
        "metadata": {
            "timestamp": "2025-04-06T12:00:00Z",
            "session_id": "123e4567-e89b-12d3-a456-426614174000"
        },
        "tags": ["Greeting"]
    }
    response = client.post("/chat/", json=missing_user_id_data)
    assert response.status_code == 422  # Expecting validation error for missing user_id
    assert "user_id" in str(response.json())  # Check if error is related to user_id

# New Code: Test for missing metadata
@pytest.mark.asyncio
async def test_chat_missing_metadata():
    missing_metadata_data = {
        "user_id": "alice",
        "text": "Hello",
        "tags": ["Greeting"]
    }
    response = client.post("/chat/", json=missing_metadata_data)
    assert response.status_code == 422  # Expecting validation error for missing metadata
    assert "metadata" in str(response.json())  # Check if error is related to metadata

# New Code: Test for very long user_id
@pytest.mark.asyncio
async def test_chat_long_user_id():
    long_user_id_data = {
        "user_id": "a" * 1000,  # Very long user_id
        "text": "Hello",
        "metadata": {
            "timestamp": "2025-04-06T12:00:00Z",
            "session_id": "123e4567-e89b-12d3-a456-426614174000"
        },
        "tags": ["Greeting"]
    }
    response = client.post("/chat/", json=long_user_id_data)
    assert response.status_code == 200  # Should still work
    assert response.json()["user_id"] == "a" * 1000    