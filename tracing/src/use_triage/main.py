import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, trace
from agents.run import RunConfig
from pydantic import BaseModel
import asyncio
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()
import agentops


# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

agentops_api_key = os.getenv("AGENTOPS_API_KEY")

if not agentops_api_key:
    raise ValueError("AGENTOPS_API_KEY is not set. Please check your .env file.")

agentops.init(api_key=agentops_api_key)


# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)



gemini_api_key =os.environ["GEMINI_API_KEY"]

if not gemini_api_key:
    raise "there is no api key"

class HomeworkOuput(BaseModel):
    is_homework: bool
    reasoning: str


math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math quistions",
    instructions="You provide help with math problems. Explain your reasoning at each step include example",
    model=model
)    

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical quistion",
    instructions="You provide assistance with historical queries. Explain important  events and context clearly",
    model=model,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework quistion",
    model=model,
    handoffs=[history_tutor_agent, math_tutor_agent],
)

@cl.on_message
async def main(message):
    # Send an initial "thinking..." message
    thinking_msg = await cl.Message(content="🤔 Thinking...").send()

    with trace("Handoffs"):
        output = await Runner.run(triage_agent, message.content)

    # Update the previous message with the final response
    await cl.Message(content=output.final_output).send()

    # session.end_session("Success")    

if __name__ == "__main__":
    
    asyncio.run(main())
