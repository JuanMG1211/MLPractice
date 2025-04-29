import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_chauvenets_removed_outliers.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()

# Visualization of missing values
subset = df[df["set"] == 82]["gyro_z"].plot()

# Using linear interpolation we can assing values
# to missing data, as most ML models don't handle
# null values too well, and dropping rows might make
# us lose a lot of data, reducing the accuracy
# of the future model.
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()  # No more missing values

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

# Set duration = Last repetition time - Initial Repetition Time
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

# Loop to calculate average duration

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# Average duration of heavy and medium sets each
duration_df = df.groupby(["category"])["duration"].mean()

# Average duration of a single heavy set repetition
duration_df.iloc[0] / 5

# Average duration of a single medium set repetition
duration_df.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# Eliminating noise with a modifiable frequency that keeps
# anything that's below it and removes anything that's above it
# hence the name "lowpass"
df_lowpass = df.copy()
LowPass = LowPassFilter()

sample = 1000 / 200  # 5 entries per second, one every 200ms
# Higher number means smoother data (not always better)
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", sample, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 30]
print(subset["label"][0])

# Visual difference between raw data and filtered data
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Filtering all data columns and replacing original values
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sample, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# Reducing data complexity by transforming data into bigger
# components, with fewer variables that capture the most
# ammount of information from the original dataset.

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# Elbow technique. An increasing number of components
# is tested to see the variance each one causes. In this
# case, 3 seems to be the optimal number of Principal Components.
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 30]
subset[["pca_1", "pca_2", "pca_3"]].plot()
# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# The advantage of Sum of Squares is it's impartiality to device orientation

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyro_r = (
    df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2
)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

subset = df_squared[df_squared["set"] == 30]
subset[["acc_r", "gyro_r"]].plot(subplots=True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyro_r"]

# Computing dataset rolling mean and rolling std for more data
# to train the model. More insight into data patterns and smoothing
# out small fluctuations.

# 1 second window size
window_size = int(1000 / 200)

# Looking back in dataset, involving data from different
# excercises together (MISTAKE)
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "std")

# In order to avoid this, Temporal Abstraction
# calculations must be made separated by excercise
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "std")
    df_temporal_list.append(subset)


df_temporal = pd.concat(df_temporal_list)

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyro_y", "gyro_y_temp_mean_ws_5", "gyro_y_temp_std_ws_5"]].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

sampling_rate = int(1000 / 200)  # 5 measurements per second
window_size = int(2900 / 200)  # Average repetition duration

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], window_size, sampling_rate)

subset = df_freq[df_freq["set"] == 6]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# Calculate by set to prevent combining data from different sets
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(
        subset, predictor_columns, window_size, sampling_rate
    )
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for label in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == label]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=label)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/02_data_features.pkl")
