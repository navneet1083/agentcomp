import autogen
import time


class AutoGenAgent:
    # def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name

        config_list = [{
            "model": self.model_name,
            "api_key": self.openai_api_key
        }]

        llm_config = {
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
            "temperature": 0,
        }

        self.autogen_agent = autogen.AssistantAgent(name="assistant", llm_config=llm_config)
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            code_execution_config={"work_dir": "autogen_output", "use_docker": False},
            llm_config=llm_config,
            system_message=(
                "Reply TERMINATE if the task is fully completed. "
                "Otherwise, reply CONTINUE or explain why the task is not yet solved."
            ),
        )

    def execute_task(self, prompt):
        start_time = time.time()

        response = self.user_proxy.initiate_chat(
            self.autogen_agent,
            message={"role": "user", "content": prompt},
        )

        content = response.chat_history[-1]["content"]
        time_taken = time.time() - start_time
        return content, time_taken

# class AutoGenAgent:
#     def __init__(self, openai_api_key):
#         self.openai_api_key = openai_api_key
#         config_list = [{"model": "gpt-4", "api_key": self.openai_api_key}]
#         llm_config = {
#             "timeout": 600,
#             "cache_seed": 42,
#             "config_list": config_list,
#             "temperature": 0,
#         }
#         self.autogen_agent = autogen.AssistantAgent(name="assistant", llm_config=llm_config)
#         self.user_proxy = autogen.UserProxyAgent(
#             name="user_proxy",
#             human_input_mode="NEVER",
#             max_consecutive_auto_reply=10,
#             is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
#             code_execution_config={"work_dir": "web", "use_docker": False},
#             llm_config=llm_config,
#             system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
#         )
#
#     def execute_task(self, prompt):
#         start_time = time.time()
#         response = self.user_proxy.initiate_chat(
#             self.autogen_agent,
#             message={"role": "user", "content": prompt},
#         )
#         content = response.chat_history[-1]["content"]
#         time_taken = time.time() - start_time
#         return content, time_taken