import matplotlib.pyplot as plt


def compare_real_and_simulated_counts_in_4q_ibm_noise_model():
    import matplotlib.pyplot as plt

    real_counts = {
        "0101": 455, "0001": 405, "0100": 743, "1001": 315, "0111": 426, "0010": 476,
        "0011": 346, "1011": 591, "1010": 466, "0110": 628, "1000": 443, "1110": 528,
        "1101": 713, "0000": 754, "1111": 278, "1100": 625
    }

    simulated_counts = {
        "0000": 486, "1000": 546, "0111": 528, "0010": 541, "1001": 493, "0011": 537,
        "1100": 501, "0101": 526, "1110": 496, "0110": 521, "1010": 461, "0100": 523,
        "0001": 498, "1011": 479, "1101": 547, "1111": 509
    }

    real_strings, real_values = zip(*real_counts.items())
    simulated_strings, simulated_values = zip(*simulated_counts.items())

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(range(len(real_strings)), real_values, color='maroon', alpha=0.7, label='Real Counts')

    ax.bar([x + 0.3 for x in range(len(simulated_strings))], simulated_values, color='navy', alpha=0.7,
           label='Simulated Counts')

    ax.set_xticks(range(len(real_strings)))
    ax.set_xticklabels(real_strings, rotation=45)
    ax.set_xlabel('Bit Strings')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of Real and Simulated Counts')

    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_real_and_simulated_counts_in_4q_ibm_noise_model()
    