import os
import time
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import convert_to_openai_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


class LangGraphAgent:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.model = ChatOpenAI(api_key=openai_api_key, model=model_name)
        self.checkpointer = MemorySaver()

        def call_llm_agent(state: MessagesState):
            messages = state["messages"]
            response = self.model.invoke(messages)
            return {"messages": messages + [AIMessage(content=response.content)]}

        builder = StateGraph(MessagesState)
        builder.add_node("call_llm_agent", call_llm_agent)
        builder.set_entry_point("call_llm_agent")
        builder.add_edge(START, "call_llm_agent")

        self.graph = builder.compile(checkpointer=self.checkpointer)

    def execute_task(self, prompt, thread_id="default-thread"):
        start_time = time.time()

        input_data = {"messages": [HumanMessage(content=prompt)]}

        # Provide the required 'thread_id' in the config
        config = {"configurable": {"thread_id": thread_id}}

        response = self.graph.invoke(input_data, config=config)

        # Fetch the last AI-generated message correctly
        response_content = response["messages"][-1].content

        time_taken = time.time() - start_time

        return response_content, time_taken


