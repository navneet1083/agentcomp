from crewai import Crew, Agent, Task
import time
import os
from langchain_openai import ChatOpenAI


class CrewAIAgent:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        self.agent = Agent(
            role="assistant",
            goal="Accurately complete tasks provided by the user.",
            backstory="You are a highly capable AI assistant.",
            llm=ChatOpenAI(api_key=openai_api_key, model=model_name),
            verbose=True
        )

    def execute_task(self, prompt):
        start_time = time.time()

        task = Task(
            description=prompt,
            expected_output="Comprehensive, accurate, and relevant response.",
            agent=self.agent
        )

        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )

        response = crew.kickoff()
        time_taken = time.time() - start_time
        return response, time_taken

