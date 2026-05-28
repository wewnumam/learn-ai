# DailyBriefEngine Crew

Welcome to the DailyBriefEngine Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/daily_brief_engine/config/agents.yaml` to define your agents
- Modify `src/daily_brief_engine/config/tasks.yaml` to define your tasks
- Modify `src/daily_brief_engine/crew.py` to add your own logic, tools and specific args
- Modify `src/daily_brief_engine/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the daily-brief-engine Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The daily-brief-engine Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the DailyBriefEngine Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.

## OpenRouter Setup (optional)

If you prefer to use OpenRouter as your LLM provider, follow these steps:

1. Create an account at https://openrouter.ai and copy your API key from your account dashboard.
2. Add the following entries to the project's `.env` file (do not commit real keys):

```
OPENROUTER_API_KEY=or-<your_openrouter_key>
OPENROUTER_API_BASE=https://api.openrouter.ai/v1
MODEL=openrouter/gpt-4o
```

3. Update `src/daily_brief_engine/crew.py` (or where you configure agents) to use the `MODEL` env var or reference OpenRouter directly. Example:

```python
import os
from crewai import Agent

model = os.getenv("MODEL", "openrouter/gpt-4o")

agent = Agent(
	role="Researcher",
	goal="Research the latest AI developments",
	backstory="Expert researcher",
	llm=model,
	verbose=True,
)
```

4. On Windows PowerShell, set the env vars for the current session before running:

```powershell
$env:OPENROUTER_API_KEY = "or-<your_openrouter_key>"
$env:OPENROUTER_API_BASE = "https://api.openrouter.ai/v1"
$env:MODEL = "openrouter/gpt-4o"
crewai run
```

5. For testing an LLM call quickly, you can run a tiny Python snippet:

```python
from crewai import LLM
import os

llm = LLM(model=os.getenv("MODEL"))
print(llm("Say hello"))
```

Notes:
- Keep your API keys secret and do not commit `.env` to version control.
- If `crewai` needs extra configuration for OpenRouter, consult https://docs.crewai.com and OpenRouter docs.

