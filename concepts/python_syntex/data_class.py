# Import necessary modules for data classes and type annotations
from dataclasses import dataclass
from typing import ClassVar

# Define a data class 'American' to represent a person with some national characteristics
@dataclass
class American:
    # Class variables (shared by all instances) for national attributes
    national_language: ClassVar[str] = "English"
    national_food: ClassVar[str] = "Hamburger"
    normal_body_temperature: ClassVar[float] = 98.6
    
    # Instance variables (unique to each person)
    name: str
    age: int
    weight: float
    liked_food: str

    # Method for simulating speech; uses the class variable for language
    def speaks(self):
        return f"{self.name} is speaking... {American.national_language}"

    # Method for simulating eating action
    def eats(self):
        return f"{self.name} is eating..."

    # Static method to return the national language without needing an instance
    @staticmethod
    def country_language():
        return American.national_language

# Calling the static method to show the national language
print(American.country_language())

# Create an instance 'john' of the American class
john = American(name="John", age=25, weight=65, liked_food="P")

# Print the result of the speaks method for 'john'
print(john.speaks())

# Print the result of the eats method for 'john'
print(john.eats())

# Display the complete 'john' instance showing all its attributes
print(john)

# Print individual attributes of 'john'
print(john.name)
print(john.age)
print(john.weight)

# Print the class variables (national characteristics) from the American class
print(American.national_language)
print(American.national_food)
