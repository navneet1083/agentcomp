import time
import importlib
import os


class BaseAgentFramework:
    def execute_task(self, prompt):
        raise NotImplementedError("This method should be implemented by subclasses.")


class AgentFactory:
    @staticmethod
    def create_agent(agent_name, **kwargs) -> BaseAgentFramework:
        """
        Dynamically load and instantiate agent classes from agents_deck.

        Parameters:
            agent_name (str): Name of the agent module/class (e.g., 'LangGraphAgent').
            kwargs: Initialization parameters (like api_key).

        Returns:
            Instance of the requested agent class.
        """
        module_name = f"agents_deck.{agent_name.lower()}"
        class_name = agent_name

        try:
            agent_module = importlib.import_module(module_name)
            agent_class = getattr(agent_module, class_name)
            return agent_class(**kwargs)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Error loading {agent_name}: {str(e)}")

# class BaseAgentFramework:
#     def execute_task(self, prompt):
#         raise NotImplementedError("Execute task method must be implemented.")
#
# class LangGraphAgent(BaseAgentFramework):
#     def execute_task(self, prompt):
#         start_time = time.time()
#         # Placeholder for LangGraph task execution
#         output = f"LangGraph output for '{prompt}'"
#         time_taken = time.time() - start_time
#         return output, time_taken
#
# class AutoGenAgent(BaseAgentFramework):
#     def execute_task(self, prompt):
#         start_time = time.time()
#         # Placeholder for AutoGen task execution
#         output = f"AutoGen output for '{prompt}'"
#         time_taken = time.time() - start_time
#         return output, time_taken
#
# class CrewAIAgent(BaseAgentFramework):
#     def execute_task(self, prompt):
#         start_time = time.time()
#         # Placeholder for CrewAI task execution
#         output = f"CrewAI output for '{prompt}'"
#         time_taken = time.time() - start_time
#         return output, time_taken
#
# # class OpenAISwarmAgent(BaseAgentFramework):
# #     def execute_task(self, prompt):
# #         start_time = time.time()
# #         # Placeholder for OpenAI Swarm task execution
# #         output = f"Swarm output for '{prompt}'"
# #         time_taken = time.time() - start_time
# #         return output, time_taken
