<<<<<<< HEAD
from agents import Agent, Runner, ModelSettings
from waither_assistent.weather_tool import get_weather
import asyncio


weather_instructions = """
You are WeatherWise, a specialized assistant for explaining weather concepts and conditions.
When a user asks for weather info, call the get_weather tool.
Note: You cannot provide real-time forecasts beyond what get_weather returns.
"""


weather_agent = Agent(
    name = "WeatherWise",
    instructions= weather_instructions,
    model="gpt-3.5-turbo",
    model_settings=ModelSettings(temperature=0.5, max_tokens=256),
    tools= [get_weather]
)


#async def test():
#    result = await Runner.run(weather_agent, "What's the weather like in London?")
#    print(result.final_output)

=======
from agents import Agent, Runner, ModelSettings
from waither_assistent.weather_tool import get_weather
import asyncio


weather_instructions = """
You are WeatherWise, a specialized assistant for explaining weather concepts and conditions.
When a user asks for weather info, call the get_weather tool.
Note: You cannot provide real-time forecasts beyond what get_weather returns.
"""


weather_agent = Agent(
    name = "WeatherWise",
    instructions= weather_instructions,
    model="gpt-3.5-turbo",
    model_settings=ModelSettings(temperature=0.5, max_tokens=256),
    tools= [get_weather]
)


#async def test():
#    result = await Runner.run(weather_agent, "What's the weather like in London?")
#    print(result.final_output)

>>>>>>> eaf0fb253bb1cd5503ac3577d45e2a761e79f169
#asyncio.run(test())