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






# 📜 Custom Hooks for Logging
class LoggingHooks(AgentHooks):
    async def on_agent_start(self, agent: Agent, input: str) -> None:
        print(f"Agent {agent.name} started with input: {input}")

    async def on_agent_complete(self, agent: Agent, output: str) -> None:
        print(f"Agent {agent.name} completed with output: {output}")

# 🤖 Create an Agent with Hooks
hooked_agent = Agent(
    name="Hooked Agent",
    instructions="Respond normally",
    model="o3-mini",
    hooks=LoggingHooks(),
)

# 🎭 Chainlit Event (Start Chat)
@cl.on_message
async def main(message: cl.Message):
    try:
        # Run the agent with user input
