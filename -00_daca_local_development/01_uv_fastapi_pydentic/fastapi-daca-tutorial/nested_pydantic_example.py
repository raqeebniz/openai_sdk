from pydantic import BaseModel, EmailStr,  field_validator,  ValidationError
from typing import List


# Define a nested model
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class UserWithAddress(BaseModel):
    id: int
    name: str
    email: EmailStr    # Built-in validator for email format
    addresses: List[Address]    # List of nested Address models


    @field_validator("name")
    def name_must_be_at_least_two_chars(cls, v):
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters long")
        return v

# Test with invalid data
try:
    invalid_user = UserWithAddress(
        id=3,
        name="A",  # Too short . instead A use full name for valid data 
        email="charlie@example.com",
        addresses=[{"street": "789 Pine Rd", "city": "Chicago", "zip_code": "60601"}],
    )
except ValidationError as e:
    print(e)

print(invalid_user)    