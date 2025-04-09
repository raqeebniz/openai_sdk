from agents import (AsyncOpenAI, OpenAIChatCompletionsModel, 
                set_default_openai_client, set_tracing_disabled, 
                Agent, RunContextWrapper, RunHooks, Runner, 
                Tool, Usage, function_tool
                )
from dataclasses import dataclass
from typing import Any
import asyncio
import random
import os




gemini_api_key = os.environ.get("GEMINI_API_KEY")


# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

set_default_openai_client(external_client)
set_tracing_disabled(True)



@dataclass
class TestHooks(RunHooks):
    event_counter = 0
    name = "TestHooks"

    async def on_agent_start(self, context:RunContextWrapper, agent:Agent) -> None:
        self.event_counter += 1
        print(f'### {self.name} {self.event_counter}: Agent {agent.name} started. Usage: {context.usage}')

    async def on_agent_end(self, context:RunContextWrapper, agent:Agent, output:Any) -> Any:
        self.event_counter += 1
        print(f"### {self.name} {self.event_counter}: Agent {agent.name} ended. Usage: {context.usage}, Output: {output}")


start_hook = TestHooks()

start_agent = Agent(
    name="Content Moderator Agent",
    instructions="You are content moderation agent. Watch social media content received and flag queries that need help or answer. We will answer anything about AI?",
    model=model
)


async def main():
    result = await Runner.run(start_agent,
    hooks = start_hook,
    input =f"<tweet>Will Agentic AI Die at end of 2025?.</tweet>"
    )
    print(result.final_output)

asyncio.run(main())
print("--end--")