import matplotlib.pyplot as plt

def compare_real_and_simulated_counts_in_2q_ibm_noise_model():
    real_counts = {
        "10": 2211,
        "11": 1862,
        "00": 2053,
        "01": 2066
    }

    simulated_counts = {
        "00": 2110,
        "01": 2112,
        "10": 1991,
        "11": 1979
    }

    real_strings, real_values = zip(*real_counts.items())
    simulated_strings, simulated_values = zip(*simulated_counts.items())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(range(len(real_strings)), real_values, color='maroon', alpha=0.6, label='Real Counts')

    ax.bar([x + 0.3 for x in range(len(simulated_strings))], simulated_values, color='navy', alpha=0.7,
           label='Simulated Counts')

    ax.set_xticks(range(len(real_strings)))
    ax.set_xticklabels(real_strings)
    ax.set_xlabel('Bit Strings')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of Real and Simulated Counts')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_real_and_simulated_counts_in_2q_ibm_noise_model()
