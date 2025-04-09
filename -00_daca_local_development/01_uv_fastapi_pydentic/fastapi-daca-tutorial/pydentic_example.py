from pydantic import BaseModel, ValidationError


# Define a simple model
class User(BaseModel):
    id: int
    name: str
    email: str
    age: int | None = None  # Optional field with default None

# Valid data
user_data = {"id": 1, "name": "raqeeb", "email": "raqeeb@gmail.com", "age": 21}
user = User(**user_data)

print(user)  

# print(user.dict()) ⚠️ gets the warning use model_dump() instead of dict()

print(user.model_dump()) 


# Invalid data (will raise an error)
try:
    invalid_user = User(id=5, name="Bob", email="bob@example.com")
except ValidationError as e:
    print(e)

#print(invalid_user)    