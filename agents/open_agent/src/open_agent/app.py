import os
import chainlit as cl
from dataclasses import dataclass
from typing import List
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



# 🏷️ Define user context
@dataclass
class UserContext:
    uid: str
    is_pro_user: bool

    async def fetch_purchases(self) -> List[str]:
        return ["item1", "item2"]  # Mock purchases

# 🛠️ Define a tool (Fetch user info)
@function_tool
def get_user_info(context: UserContext) -> str:
    return f"User {context.uid} is {'pro' if context.is_pro_user else 'standard'}"

# 🤖 Create User Agent with typed context
user_agent = Agent[UserContext](
    name="User Agent",
    instructions="Provide user-specific responses. Use 'get_user_info' when needed.",
    model="o3-mini",
    tools=[get_user_info],
)

# 🎭 Chainlit Event (Start Chat)
@cl.on_message
async def main(message: cl.Message):
    context = UserContext(uid="123", is_pro_user=True)  # Mock context for now

    try:
        # Run the agent with user input and context
        response = await Runner.run(user_agent, message.content, context=context)

        # Send response back to Chainlit UI
        await cl.Message(content=response.final_output).send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
