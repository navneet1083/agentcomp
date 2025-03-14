from tools.agents import AgentFactory
from inputs import test_inputs
import os

# Set your OpenAI API key securely (consider using dotenv)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = ""

agent_names = [
    # "LangGraphAgent",
    # "AutoGenAgent",
    "CrewAIAgent",
    # "OpenAISwarmAgent"
]

for agent_name in agent_names:
    print(f"\nRunning tasks using {agent_name}...")

    # Instantiate agent dynamically
    agent = AgentFactory.create_agent(agent_name, openai_api_key=OPENAI_API_KEY)

    for task in test_inputs:
        print(f"\nRunning task id {task['id']} :: {task['task']}...")
        output, exec_time = agent.execute_task(task["task"])
        # print(f"Task ID {task['id']}: Output: {output[:80]}... | Time: {exec_time:.2f}s")
        print(f"Task ID {task['id']}: Output: {output} | Time: {exec_time:.2f}s")
        print('----'*50)
        # break
