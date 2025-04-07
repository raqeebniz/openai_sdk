# Getting Started with FastAPI, UV, and Pydantic

Welcome to the first tutorial in our Dapr Agentic Cloud Ascent (DACA) series! In this baby step, we'll set up a FastAPI project using the uv Python dependency manager, dive deep into FastAPI and Pydantic, and build a more robust API with unit tests, CORS middleware, and complex data models. FastAPI will serve as the REST API layer for our agentic AI system, enabling communication between users, agents, and microservices. Let's get started!

## What You'll Learn
- How to install and use the uv Python dependency manager.
- Setting up a FastAPI project with uv.
- Understanding FastAPI's key features: automatic documentation, async support, and Pydantic integration.
- A deep dive into Pydantic for data validation and serialization.
- Building a FastAPI application with complex Pydantic models.
- Adding CORS middleware for cross-origin requests.
- Writing unit tests with pytest to ensure API reliability.
- Testing and running the API with practical examples.

## Prerequisites
- Python 3.12+ installed on your system.
- Basic familiarity with Python, command-line tools, and REST APIs.
- A code editor (e.g., VS Code).

## Step 1: Introduction to UV Python Dependency Manager

### What is UV?
uv is a modern, fast, and lightweight Python dependency manager built by the team at Astral. It's designed to replace tools like pip and virtualenv by providing a unified, high-performance solution for managing Python projects. Key features include:

- Speed: Blazing fast dependency resolution and installation (written in Rust).
- Unified Workflow: Combines dependency management, virtual environment creation, and project setup.
- Locking: Generates a uv.lock file for reproducible builds.
- Modern Features: Supports PEP 582 (no need to activate virtualenvs manually in supported environments).

uv is ideal for DACA projects as it streamlines dependency management for FastAPI, Dapr, and other components.

### Installing UV
**On macOS/Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
On Windows (PowerShell)

powershell
Copy
Edit
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Verify Installation
bash
Copy
Edit
uv --version
You should see output like uv 0.4.18 (or the latest version).

Step 2: Setting Up a FastAPI Project with UV
Create a Project Directory
bash
Copy
Edit
mkdir fastapi-daca-tutorial
cd fastapi-daca-tutorial
Initialize a Python Project with UV
bash
Copy
Edit
uv init
This creates:

A pyproject.toml file for project metadata and dependencies.

A virtual environment (.venv).

Activate the Virtual Environment
On macOS/Linux:

bash
Copy
Edit
source .venv/bin/activate
On Windows:

bash
Copy
Edit
.venv\Scripts\activate
Note: With PEP 582 support (Python 3.11+), uv may not require manual activation for running commands.

Add Dependencies
We'll need FastAPI, Uvicorn (ASGI server), and additional packages for testing:

bash
Copy
Edit
uv add fastapi uvicorn pytest pytest-asyncio httpx
This updates pyproject.toml:

toml
Copy
Edit
[project]
name = "fastapi-daca-tutorial"
version = "0.1.0"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.6",
    "pytest>=8.3.3",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.2",
]
Verify the installed dependencies:
bash
Copy
Edit
uv pip list
Step 3: Deep Dive into Pydantic
What is Pydantic?
Pydantic is a data validation and settings management library that uses Python type annotations to define and validate data schemas. It's a core dependency of FastAPI, used for request/response validation, serialization, and deserialization. Pydantic ensures type safety and provides automatic error handling, making it ideal for DACA's agentic workflows where data integrity is critical.

Key Features of Pydantic
Type-Safe Validation: Validates data against Python type hints (e.g., str, int, List[str]).

Automatic Conversion: Converts data to the correct type (e.g., string "123" to int 123).

Error Handling: Raises detailed validation errors for invalid data.

Nested Models: Supports complex, nested data structures.

Serialization: Converts models to JSON (or other formats) for API responses.

Default Values and Optional Fields: Simplifies schema definitions.

Custom Validators: Allows custom validation logic.

Getting Started with Pydantic
Let's explore Pydantic with examples before integrating it into our FastAPI app.

Basic Pydantic Model
Create a file named pydantic_examples.py:

python
Copy
Edit
from pydantic import BaseModel, ValidationError

# Define a simple model
class User(BaseModel):
    id: int
    name: str
    email: str
    age: int | None = None  # Optional field with default None

# Valid data
user_data = {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 25}
user = User(**user_data)
print(user)  # id=1 name='Alice' email='alice@example.com' age=25
print(user.dict())  # {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 25}

# Invalid data (will raise an error)
try:
    invalid_user = User(id="not_an_int", name="Bob", email="bob@example.com")
except ValidationError as e:
    print(e)
Run the script:

bash
Copy
Edit
uv run python pydantic_examples.py
Nested Models
Pydantic supports nested structures, which we'll use in our FastAPI app. Extend pydantic_examples.py:

python
Copy
Edit
from pydantic import BaseModel, EmailStr
from typing import List

# Define a nested model
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class UserWithAddress(BaseModel):
    id: int
    name: str
    email: EmailStr  # Built-in validator for email format
    addresses: List[Address]  # List of nested Address models

# Valid data with nested structure
user_data = {
    "id": 2,
    "name": "Bob",
    "email": "bob@example.com",
    "addresses": [
        {"street": "123 Main St", "city": "New York", "zip_code": "10001"},
        {"street": "456 Oak Ave", "city": "Los Angeles", "zip_code": "90001"},
    ],
}
user = UserWithAddress(**user_data)
print(user.dict())
Custom Validators
Add a custom validator to ensure the user's name is at least 2 characters long:

python
Copy
Edit
from pydantic import BaseModel, EmailStr, validator
from typing import List

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class UserWithAddress(BaseModel):
    id: int
    name: str
    email: EmailStr
    addresses: List[Address]

    @validator("name")
    def name_must_be_at_least_two_chars(cls, v):
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters long")
        return v

# Test with invalid data
try:
    invalid_user = UserWithAddress(
        id=3,
        name="A",  # Too short
        email="charlie@example.com",
        addresses=[{"street": "789 Pine Rd", "city": "Chicago", "zip_code": "60601"}],
    )
except ValidationError as e:
    print(e)
Advanced Features
Field Aliases:
Map JSON keys to Python attributes (e.g., email_address in JSON to email in Python):

python
Copy
Edit
class UserAlias(BaseModel):
    email: str = Field(..., alias="email_address")

user = UserAlias(**{"email_address": "dave@example.com"})
print(user.email)  # dave@example.com
Default Factories:
Use a function to set default values:

python
Copy
Edit
from pydantic import Field
from uuid import uuid4

class UserWithUUID(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid4()))

user = UserWithUUID()
print(user.user_id)  # e.g., "123e4567-e89b-12d3-a456-426614174000"
Validation Modes:
Configure strictness (e.g., strict=True to disable type coercion):

python
Copy
Edit
class StrictUser(BaseModel, strict=True):
    id: int

try:
    user = StrictUser(id="123")  # Will fail because "123" is a string
except ValidationError as e:
    print(e)  # type_error.integer
Why Pydantic for DACA?
Pydantic is critical for DACA because:

Data Integrity: Ensures incoming user data and agent responses are valid and type-safe.

Complex Workflows










