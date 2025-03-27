import os
import asyncio
from dotenv import load_dotenv
from typing import cast
from agents.run import RunConfig
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_key
from stripe_agent_toolkit.openai.toolkit import StripeAgentToolkit


# Load the environment variables from the .env file
load_dotenv()

set_default_openai_key("OPENAI_API_KEY")


stripe_agent_toolkit = StripeAgentToolkit(
    secret_key=os.getenv("STRIPE_SECRET_KEY"),
    configuration={
        "actions": {
            "payment_links": {
                "create": True,
            },
            "products": {
                "create": True,
            },
            "prices": {
                "create": True,
            },
        }
    },
)



agent = Agent(
    name= "Stripe Agent",
    instructions= "Integrate with Stripe effectively to support business needs.",
    tools=stripe_agent_toolkit.get_tools(),
)


@cl.on_chat_start
async def start():
    cl.user_session.set("agent", agent)
    await cl.Message(content= "Welcome to payment agents").send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content= "Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent")) 

    result = await Runner.run(agent, message.content)

    response_content = result.final_output
    msg.content = response_content

    await msg.update()