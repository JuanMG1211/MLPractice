import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"] == 1]  # Dataset subset

plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# Plot first 100 entries
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# Plot Style
mpl.style.use("seaborn-v0_8-deep")
# Plot Size
mpl.rcParams["figure.figsize"] = (20, 5)
# Plot resolution on export
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
# Another way of making subsets
query_df1 = df.query("label == 'squat'")
query_df2 = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig, ax = plt.subplots()
query_df2.groupby(["category"])["acc_y"].plot()
ax.set_ylabel = "acc_y"
ax.set_xlabel = "samples"
plt.legend()

# Acceleration on medium sets is higher than heavy sets. Makes sense. Less weight

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
df["label"].unique()
fig, ax = plt.subplots()

# Squatting Comparison

squat_df = df.query("label == 'squat'").sort_values("participant").reset_index()

squat_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# Bench Press Comparison
bench_df = df.query("label == 'bench'").sort_values("participant").reset_index()

bench_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# Overhead Press Comparison
ohp_df = df.query("label == 'ohp'").sort_values("participant").reset_index()

ohp_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# Deadlift Comparison
dead_df = df.query("label == 'dead'").sort_values("participant").reset_index()

dead_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# Standing Row Comparison
row_df = df.query("label == 'row'").sort_values("participant").reset_index()

row_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# Resting Comparison
rest_df = df.query("label == 'rest'").sort_values("participant").reset_index()

rest_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
all_axis_of = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_of[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel = "acc_y"
ax.set_xlabel = "samples"
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_of = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # Eliminate empty plots for accelerometer data
        if len(all_axis_of) > 0:
            fig, ax = plt.subplots()
            all_axis_of[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel = "acc_y"
            ax.set_xlabel = "samples"
            plt.title(f"{label} (Participant {participant})".title())
            plt.legend()

for label in labels:
    for participant in participants:
        all_axis_of = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # Eliminate empty plots for gyroscope data
        if len(all_axis_of) > 0:
            fig, ax = plt.subplots()
            all_axis_of[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
            ax.set_ylabel = "gyro_y"
            ax.set_xlabel = "samples"
            plt.title(f"{label} (Participant {participant})".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_of = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_of[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_of[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")
# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

for label in labels:
    for participant in participants:
        combined_plot_of = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # Eliminate empty plots
        if len(combined_plot_of) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_of[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_of[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper right",
                bbox_to_anchor=(1, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper right",
                bbox_to_anchor=(1, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")

            plt.title(
                f"{label} (Participant {participant})".title(), loc="center", y=2.25
            )

            plt.savefig(f"../../reports/figures/{label.title()}_({participant}).png")

            plt.show()
