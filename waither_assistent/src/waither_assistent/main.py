import os 
import asyncio 
from dotenv import load_dotenv
import chainlit as cl
from agents import Runner
from waither_assistent.weather_agent import weather_agent


load_dotenv()

@cl.on_chat_start
async def on_start():

    cl.user_session.set("chat_history", [])

    cl.user_session.set("agent", weather_agent)

    await cl.Message(content="Welcome to WeatherWise! Ask me about weather in any city.").send()



@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="Thinking...")
    await msg.send()


    # Retrieve the weather agent and existing chat history from the session
    agent = cl.user_session.get("agent")
    history = cl.user_session.get("chat history") or []
    history.append({"role":"user", "content": message.content})

    try:
        # Run Agent asynchronously with the chat history
        result = await Runner.run(agent, history)
        response_content = result.final_output

        # Update the message with the agent's response
        msg.content= response_content
        await msg.update()

        # Update the session's chat history
        cl.user_session.set("chat_history", result.to_input_list())


        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")


    except Exception as e:
        msg.content = f"An error occured: {str(e)}"    
        await msg.update()
        print(f"Error: {str(e)}")




 