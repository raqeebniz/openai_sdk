import os
import chainlit as cl
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file."
    )

# Step 1: Provider
provider =  AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
) 
 
# Step 2: 
model = OpenAIChatCompletionsModel(
    model="gemini/gemini-1.5-flash",
    openai_client=provider
)

# Step 3: Defined at run level
run_config = RunConfig(
    model=model,
    model_provider= provider,
    tracing_disabled=True 
)



# Define specialized agents
booking_agent = Agent(
    name="Booking Agent",
    instructions="Help users book flights and hotels.",
    model=model,
)

refund_agent = Agent(
    name="Refund Agent",
    instructions="Assist users with refunds and cancellations.",
    model=model
)

# Main agent that decides who should handle the request
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, handoff to the Booking Agent. "
        "If they ask about refunds, handoff to the Refund Agent."
    ),
    handoffs=[booking_agent, refund_agent],  # Linking the specialized agents
)

# Chainlit UI
@cl.on_message
async def main(message: cl.Message):
    result = await Runner.run(triage_agent, message.content)
    
    # Show response from the correct agent
    await cl.Message(content=f"🤖 Response:\n\n{result}").send()