import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Load data
calibration_scores = np.load('calibration_scores.npy').squeeze()
test_scores = np.load('test_scores.npy').squeeze()
calibration_emb = np.load('calibration_embeddings.npy').squeeze()
test_emb = np.load('test_embeddings.npy').squeeze()

# Print shapes for debugging
print("Calibration scores shape:", calibration_scores.shape)
print("Calibration embeddings shape:", calibration_emb.shape)
print("Test scores shape:", test_scores.shape)
print("Test embeddings shape:", test_emb.shape)

# Define alpha and c ranges
alpha_values = np.linspace(0.05, 0.95, 10)
c_values = np.linspace(0.05, 0.45, 9)

# Store FDR and power for plotting
fdr_results = {c: [] for c in c_values}
power_results = {c: [] for c in c_values}

fdr_min = {c: [] for c in c_values}
power_min = {c: [] for c in c_values}

fdr_max = {c: [] for c in c_values}
power_max = {c: [] for c in c_values}

# Iterate over alpha and c
for c in c_values:
    for alpha in alpha_values:
        fdr_list = []
        power_list = []

        # Run 100 independent experiments for each combination of alpha and c
        for _ in trange(500):
            # Split calibration data into training and calibration sets
            X_train, X_calib, y_train, y_calib = train_test_split(calibration_emb, calibration_scores, test_size=0.3, random_state=None)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict on calibration and test data
            y_calib_pred = model.predict(X_calib)
            y_test_pred = model.predict(test_emb)

            # Compute p-values
            p_values = []
            for d in y_test_pred:
                U_j = np.random.uniform(0, 1)  # Uniform random variable
                indicator_1 = (y_calib_pred < d) & (y_calib >= c)
                indicator_2 = (y_calib_pred == d) & (y_calib >= c)
                numerator = np.sum(indicator_1) + U_j * (1 + np.sum(indicator_2))
                p_value = numerator / (1 + len(y_calib))
                p_values.append(p_value)

            # Sort p-values and find threshold
            sorted_p_values = sorted(p_values)
            max_r = 0
            for r, p_value in enumerate(sorted_p_values):
                if p_value > (r + 1) * alpha / len(sorted_p_values):
                    break
                else:
                    max_r = r + 1

            if max_r == 0:
                filtered_indices = []
            else:
                p_threshold = sorted_p_values[max_r - 1]
                filtered_indices = [i for i, p_value in enumerate(p_values) if p_value <= p_threshold]

            # Compute FDR and power
            filtered_diff = np.array([y_test_pred[i] for i in filtered_indices])
            fdr = np.sum(filtered_diff >= c) / max(1, len(filtered_diff))
            power = np.sum(filtered_diff < c) / max(1, np.sum(y_test_pred < c))

            # Append to lists
            fdr_list.append(fdr)
            power_list.append(power)

        # Compute averages
        fdr_list = np.array(fdr_list)
        power_list = np.array(power_list)

        Q1_fdr = np.percentile(fdr_list, 25)
        Q3_fdr = np.percentile(fdr_list, 75)
        outliers_fdr = (fdr_list < Q1_fdr - 1.5 * (Q3_fdr - Q1_fdr)) | (fdr_list > Q3_fdr + 1.5 * (Q3_fdr - Q1_fdr))
        fdr_list = fdr_list[~outliers_fdr]

        Q1_power = np.percentile(power_list, 25)
        Q3_power = np.percentile(power_list, 75)
        outliers_power = (power_list < Q1_power - 1.5 * (Q3_power - Q1_power)) | (power_list > Q3_power + 1.5 * (Q3_power - Q1_power))
        power_list = power_list[~outliers_power]

        avg_fdr = np.mean(fdr_list)
        avg_power = np.mean(power_list)

        fdr_results[c].append(avg_fdr)
        power_results[c].append(avg_power)

        fdr_min[c].append(np.percentile(fdr_list, 5))
        power_min[c].append(np.percentile(power_list, 5))

        fdr_max[c].append(np.percentile(fdr_list, 95))
        power_max[c].append(np.percentile(power_list, 95))

# Plot results
for c in c_values:
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_values, fdr_results[c], label="FDR", marker="o", color='blue', markersize=8)
    plt.fill_between(alpha_values, fdr_min[c], fdr_max[c], color='blue', alpha=0.2)
    plt.plot(alpha_values, power_results[c], label="Power", marker="^", color='orange', markersize=8)
    plt.fill_between(alpha_values, power_min[c], power_max[c], color='orange', alpha=0.2)
    plt.plot([0,1], [0,1], linestyle='--', color='black', alpha=0.5) 
    plt.tick_params(axis='both', labelsize=18)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    #plt.title(f"FDR and Power vs Alpha for c={c:.2f}", fontsize=20)
    plt.xlabel(r"Target FDR level at $\alpha$", fontsize=20, fontweight='bold')
    plt.ylabel("FDR and Power", fontsize=20, fontweight='bold')
    plt.legend(fontsize=18, loc='center right')
    plt.grid()
    plt.savefig(f"FDR_c_{c:.2f}.png", bbox_inches="tight")
    #plt.show()
