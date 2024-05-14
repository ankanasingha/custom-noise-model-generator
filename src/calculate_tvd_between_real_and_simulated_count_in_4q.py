import json
import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_average_tvd():
    avg_tvd_by_iteration = []
    for i in range(1, 6):
        real_counts_file_path = os.path.join(os.path.dirname(os.getcwd()), "resources", "4q", "real_counts",
                                             f"real_counts_{i}.json")
        with open(real_counts_file_path, 'r') as file:
            real_counts = json.load(file)

        simulated_counts_file_path = os.path.join(os.path.dirname(os.getcwd()), "resources", "4q", "simulated_counts",
                                             f"simulated_counts_{i}.json")

        with open(simulated_counts_file_path, 'r') as file:
            simulated_counts = json.load(file)

        avg_tvd = np.mean([calculate_tvd(simulated_counts[i], real_counts[i]) for i in range(len(real_counts))])
        avg_tvd_by_iteration.append(avg_tvd)
        print(f"The average TVD value for set_{i} is: {avg_tvd}")

    print(f"The average TVD value across all the sets is: {sum(avg_tvd_by_iteration) / 5}")
    return avg_tvd_by_iteration


def calculate_tvd(real_count, simulated_count):
    total_real_counts = sum(real_count.values())
    real_probs = {k: v / total_real_counts for k, v in real_count.items()}

    total_simulated_counts = sum(simulated_count.values())
    simulated_probs = {k: v / total_simulated_counts for k, v in simulated_count.items()}
    return 0.5 * sum(abs(simulated_probs.get(k, 0) - real_probs.get(k, 0)) for k in set(simulated_probs)
                     | set(real_probs))


def calculate_tvd_between_real_and_simulated_count_in_4q():
    avg_tvd_by_iteration = calculate_average_tvd()

    # Labels for each set
    labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5']

    # Plotting the bar graph
    plt.bar(labels, avg_tvd_by_iteration, color='darkgoldenrod')

    # Adding title and labels
    plt.title('Average TVD Values for 5 Sets of Circuits (4q)')
    plt.xlabel('Set')
    plt.ylabel('Average TVD Value')

    # Displaying the plot
    plt.show()


if __name__ == "__main__":
    calculate_tvd_between_real_and_simulated_count_in_4q()