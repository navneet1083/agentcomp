from inputs import test_inputs, ground_truths
from tools.evaluate import evaluate_output
from tools.agents import AgentFactory
import pandas as pd


class Benchmark:
    def __init__(self, agent):
        self.agent = agent

    def run_benchmark(self):
        results = []
        total_time = 0
        total_accuracy = 0

        for task in test_inputs:
            output, exec_time = self.agent.execute_task(task['task'])
            accuracy = evaluate_output(output, ground_truths[task['id']])

            results.append({
                'Task ID': task['id'],
                'Task': task['task'],
                'Output': output,
                'Execution Time (s)': round(exec_time, 2),
                'Accuracy Score': accuracy
            })

            total_time += exec_time
            total_accuracy += accuracy

        avg_time = total_time / len(test_inputs)
        avg_accuracy = total_accuracy / len(test_inputs)
        scalability_score = round((1 / avg_time) * 100, 2)

        metrics = {
            "Average Execution Time (s)": round(avg_time, 2),
            "Average Accuracy": round(avg_accuracy, 2),
            "Scalability Score": scalability_score
        }

        return results, metrics


# Example Usage:
if __name__ == "__main__":

    OPENAI_API_KEY = ""

    agent_names = [
        "LangGraphAgent",
        # "AutoGenAgent",
        # "CrewAIAgent",
        # "OpenAISwarmAgent"
    ]

    all_metrics = []

    for agent_name in agent_names:
        print(f"\nRunning tasks using {agent_name}...")

        # Instantiate agent dynamically
        agent_instance = AgentFactory.create_agent(agent_name, openai_api_key=OPENAI_API_KEY)
        benchmark = Benchmark(agent_instance)
        results, metrics = benchmark.run_benchmark()

        print(f"\nBenchmark Results for {agent_name}:")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        print("\nSummary Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        metrics['Framework'] = agent_name
        all_metrics.append(metrics)

    # Display comparison summary
    comparison_df = pd.DataFrame(all_metrics)
    print("\n=== Comparison Summary ===")
    print(comparison_df[['Framework', 'Average Execution Time (s)', 'Average Accuracy', 'Scalability Score']].to_string(
        index=False))
