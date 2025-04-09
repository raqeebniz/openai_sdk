import os 
import requests
from dataclasses import dataclass
from agents import function_tool

@dataclass
class WeatherInfo:
    location: str
    temperature: float
    description: str



@function_tool
def get_weather(city:str) -> str:
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        data = requests.get(url).json()
        info = WeatherInfo(
            location=data["name"],
            temperature=data["main"]["temp"],
            description=data["weather"][0]["description"],
        )
        return (
            f"Weather in {info.location}: {info.temperature}°C, {info.description}."
        )
    except Exception as e:
        return f"Error fetching data: {e}"