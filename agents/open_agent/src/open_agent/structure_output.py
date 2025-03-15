import os
import chainlit as cl
from dataclasses import dataclass
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, ModelSettings, function_tool

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Provider
provider =  AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
) 
 
# Step 2: 
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=provider
)

# Step 3: Defined at run level
run_config = RunConfig(
    model=model,
    model_provider= provider,
    tracing_disabled=True 
)



# 🗓️ Define a structured format for calendar events
class CalendarEvent(BaseModel):
    name: str  # Event name
    date: str  # Event date
    participants: list[str]  # List of participants

# 🧠 Create an agent that extracts calendar events
calendar_agent = Agent(
    name="Calendar Extractor",
    instructions="Extract calendar events from text",
    model=model,
    output_type=CalendarEvent,  # 👈 Structured output
)

# 🚀 Chainlit app to interact with the agent
@cl.on_message
async def main(message: cl.Message):
    result = await Runner.run(calendar_agent,message.content)
    
    # 🛠️ Extract the structured output correctly
    event_data = result.final_output  # 👈 Extract actual CalendarEvent object

    if event_data is None:
        await cl.Message(content="❌ No event detected. Try again!").send()
    else:
        await cl.Message(content=f"📅 Event Extracted:\n\n{event_data.model_dump_json(indent=2)}").send()
