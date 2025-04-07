# Tracing in the Project

## Overview
This project demonstrates how to use tracing with AgentOps and OpenAI SDK while running an AI agent-based system. The tracing feature helps monitor and debug agent handoffs and interactions.

## Prerequisites
- Python 3.11+
- Required dependencies installed (see `requirements.txt`)
- `.env` file with `AGENTOPS_API_KEY` and `GEMINI_API_KEY`

## How to Use Tracing

1. **Ensure your API keys are set**
   - Add your API keys in a `.env` file:
     ```env
     AGENTOPS_API_KEY=your_agentops_api_key
     GEMINI_API_KEY=your_gemini_api_key
     ```

2. **Enable verbose logging (optional for debugging)**
   ```python
   from agents import enable_verbose_stdout_logging
   enable_verbose_stdout_logging()
   ```

3. **Initialize AgentOps tracing**
   ```python
   import agentops
   agentops.init()
   ```

4. **Use tracing in the main execution**
   ```python
   from agents import trace
   
   async def main():
       with trace("Handoffs"):  # Enables tracing for agent interactions
           output = await Runner.run(triage_agent, "what is 6 * 9 + 10")
           print(output.final_output)
   ```

5. **Run the script**
   ```sh
   python main.py
   ```

Tracing will now capture interactions between agents and log their execution for debugging and analysis.

