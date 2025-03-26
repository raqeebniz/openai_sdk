import chainlit as cl
from agents import Agent, RunContextWrapper, Runner
from dataclasses import dataclass

# 🏷️ Define user context
@dataclass
class UserContext:
    name: str

# 🛠️ Function to generate dynamic instructions based on user context
def dynamic_instructions(
    context: RunContextWrapper[UserContext], 
    agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."

# 🤖 Create an Agent with dynamic instructions
dynamic_agent = Agent[UserContext](
    name="Dynamic Agent",
    instructions=dynamic_instructions,
    model="o3-mini",
)

# 🎭 Chainlit Event (Start Chat)
@cl.on_message
async def main(message: cl.Message):
    context = UserContext(name="Alice")  # Mock user context

    try:
        # Run the agent with user input and context
        response = await Runner.run(dynamic_agent, message.content, context=context)

        # Send response back to Chainlit UI
        await cl.Message(content=response.final_output).send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
