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


# 🎭 User Context Model
@dataclass
class UserContext:
    user_id: str
    name: str

# 🌤️ Weather Response Model
class WeatherResponse(BaseModel):
    city: str
    condition: str
    temperature: float

# ⛅ Function Tool: Get Weather
@function_tool
def get_weather(context: UserContext, city: str) -> str:
    return f"Weather for {city} accessed by {context.name}"

# 🛠️ Custom Hooks (Logs when agent starts)
class CustomHooks(AgentHooks):
    async def on_agent_start(self, agent: Agent, input: str) -> None:
        print(f"Starting {agent.name} with: {input}")

# 🌍 Forecast Agent
forecast_agent = Agent[UserContext](
    name="Forecast Agent",
    instructions="Provide weather forecasts",
    model=model,
    tools=[get_weather],
    output_type=WeatherResponse,
)

# 🎧 Support Agent
support_agent = Agent[UserContext](
    name="Support Agent",
    instructions="Handle support queries",
    model=model,
)

# 🤖 Main Weather Assistant (Handles queries & routes requests)
main_agent = Agent[UserContext](
    name="Weather Assistant",
    instructions="Handle weather-related queries and support",
    model=model,
    tools=[get_weather],
    handoffs=[forecast_agent, support_agent],
    hooks=CustomHooks(),
)

# 🎭 Casual Version (Modified behavior)
casual_agent = main_agent.clone(
    name="Casual Weather Bot",
    instructions="Respond casually about weather",
)

# 🎭 Chainlit Event: Start Chat
@cl.on_message
async def handle_user_message(message: cl.Message):
    context = UserContext(user_id="123", name="John")  # Mock user context

    try:
        # Run the main agent
        response = await Runner.run(main_agent, message.content, context=context)

        # Send response back to Chainlit UI
        await cl.Message(content=response.final_output).send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
