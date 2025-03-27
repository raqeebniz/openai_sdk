import os
import chainlit as cl
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


# 🛠️ Define a tool (Weather function)
@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

# 🤖 Create an AI Agent
agent = Agent(
    name="Haiku agent",
    instructions=("Always respond in haiku form."
                  "If the user asks about weather, call the 'get_weather' tool."
                  ),
    model=model,
    tools=[get_weather],  # Adding weather tool
)

# 🎭 Chainlit Event (Start Chat)
@cl.on_message
async def main(message: cl.Message):
    try:
        # Run the agent with user input using Runner
        response = await Runner.run(agent, message.content)

        # Send response back to Chainlit UI
        await cl.Message(content=response.final_output).send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

