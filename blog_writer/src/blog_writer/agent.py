import os
import asyncio
from dotenv import load_dotenv, find_dotenv

# Assuming these are custom modules
try:
    from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    exit(1)

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("API key not found in environment variables")

# Setup provider and model
try:
    provider = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    
    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=provider
    )

    run_config = RunConfig(
        model=model,
        model_provider=provider,
        tracing_disabled=True
    )
except Exception as e:
    print(f"Error setting up provider/model: {e}")
    exit(1)

# Define tool
@function_tool
def complete_blog():
    """Mark the blog post as complete."""
    return "Task completed"

# Agent instructions with hardcoded topic "agentic ai"
def admin_instructions(context, input):
    return f"""You are the Admin Agent overseeing the blog post project on 'agentic ai'.
Your task is to {input}.
Your responsibilities include initiating the project, providing guidance, and reviewing the final content.
Once you've set the topic, hand off to the Planner Agent."""

def planner_instructions(context, input):
    return f"""You are the Planner Agent. Based on the task to {input} on 'agentic ai',
Organize the content into topics and sections with clear headings that will each be individually researched as points in the greater blog post.
Once the outline is ready, hand off to the Researcher Agent."""

def researcher_instructions(context, input):
    return f"""You are the Researcher Agent. Your task is to provide dense context and information to {input} on 'agentic ai' as outlined by the Planner Agent.
This research will serve as the information that will be formatted into the body of a blog post. Provide comprehensive research like notes for each section outlined.
Once your research is complete, hand off to the Writer Agent."""

def writer_instructions(context, input):
    return f"""You are the Writer Agent. Your task is to {input} on 'agentic ai' using the prior information, following the outline from the Planner Agent.
Summarize and include as much relevant information from the research into the blog post.
The blog post should be large, as the context provided is dense.
Write clear, engaging content for each section.
Once the draft is complete, hand off to the Editor Agent."""

def editor_instructions(context, input):
    return f"""You are the Editor Agent. Review and edit the blog post completed by the Writer Agent for the task to {input} on 'agentic ai'.
Make necessary corrections and improvements.
Once editing is complete, mark the blog post as complete."""

# Define agents
admin_agent = Agent(
    name="Admin Agent",
    instructions=admin_instructions,
    handoffs=["Planner Agent"],
)

planner_agent = Agent(
    name="Planner Agent",
    instructions=planner_instructions,
    handoffs=["Researcher Agent"],
)

researcher_agent = Agent(
    name="Researcher Agent",
    instructions=researcher_instructions,
    handoffs=["Writer Agent"],
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=writer_instructions,
    handoffs=["Editor Agent"],
)

editor_agent = Agent(
    name="Editor Agent",
    instructions=editor_instructions,
    tools=[complete_blog],
    handoffs=[],
)

async def main():
    try:
        result = await Runner.run(
            admin_agent,
            input="write a blog post",  # Provide the input argument
            context={}  # Empty context since topic is hardcoded
        )
        print("Final output:", result.final_output)
    except Exception as e:
        print(f"Error during execution: {e}")

# For compatibility with script runners that don't handle async
def run():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())