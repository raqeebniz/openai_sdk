import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from pydantic import BaseModel
import asyncio
from agents import enable_verbose_stdout_logging
enable_verbose_stdout_logging()


# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

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

triage_agent = Agent (
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework quistion",
    model=model,
    handoffs=[history_tutor_agent, math_tutor_agent],
)

async def main():
    output = await Runner.run(triage_agent, "what is 6 * 9 + 10")
    print(output.final_output)
    

if __name__ == "__main__":
    asyncio.run(main())    