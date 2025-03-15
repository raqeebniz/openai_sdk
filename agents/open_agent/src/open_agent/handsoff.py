import os
import chainlit as cl
from pydantic import BaseModel
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, AgentHooks, function_tool

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


# ✈️ Booking Agent
booking_agent = Agent(
    name="Booking Agent",
    instructions="Handle booking requests.",
    model="o3-mini",
)

# 💰 Refund Agent
refund_agent = Agent(
    name="Refund Agent",
    instructions="Handle refund requests.",
    model="o3-mini",
)

# 🤖 Triage Agent (Decides which agent to hand off to)
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, handoff to the booking agent. "
        "If they ask about refunds, handoff to the refund agent."
    ),
    model="o3-mini",
    handoffs=[booking_agent, refund_agent],  # Enables automatic agent switching
)

# 🎭 Chainlit Event (Start Chat)
@cl.on_message
async def main(message: cl.Message):
    try:
        # Run the triage agent to decide the handoff
        response = await Runner.run(triage_agent, message.content)

        # Send response back to Chainlit UI
        await cl.Message(content=response.final_output).send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
