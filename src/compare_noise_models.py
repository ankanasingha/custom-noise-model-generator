import matplotlib.pyplot as plt
import numpy as np


def compare_noise_model():
    models = ['IBM\n Noise Model', 'Custom\nNoise\n Model\n(Pre-Opt)', 'Custom\nNoise\n Model\n(BO)',
              'Unified\nNoise\n Model\n(BO)', 'Custom\nNoise\n Model\n(PSO)', 'Unified\nNoise\n Model\n(PSO)']
    tvd_values = [0.185, 0.187, 0.131, 0.142, 0.146, 0.151]  # TVD values corresponding to the models

    # Set the positions and width for the bars
    positions = np.arange(len(models))
    width = 0.5  # the width of the bars

    # Create the bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(positions, tvd_values, width, align='center', alpha=0.7, color='darkgoldenrod')

    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Noise Models and Optimization Techniques')
    ax.set_ylabel('Total Variation Distance (TVD)')
    ax.set_title('Comparison of TVD Values Across Noise Models (4-Qubit Circuits)')
    ax.set_xticks(positions)
    ax.set_xticklabels(models)

    # Adding value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_noise_model()
