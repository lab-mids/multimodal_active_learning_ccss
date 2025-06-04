import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import itertools
import matplotlib.colors as mcolors
STRATEGY_DISPLAY_NAMES = {
    "Centroids_saturation_high": "Cent_sat_high",
    "Centroids_saturation_medium": "Cent_sat_med",
    "Centroids_saturation_low": "Cent_sat_low",
    "Top5Similarity": "T5S",
    "Max Comp": "Max Comp",
    "Min Comp": "Min Comp",
    "Random": "Random",
    "LHS": "LHS",
    "K-Means": "K-Means",
    "Farthest": "FPS",
    "K-Center": "K-Center",
    "ODAL": "ODAL"
}
base_strategies = [
    "Top5Similarity", "Max Comp", "Min Comp", 
    "Centroids_saturation_high", "Centroids_saturation_medium", "Centroids_saturation_low",
    "Random", "LHS", "K-Means", "Farthest", "K-Center", "ODAL"
]

def plot_strategy_across_datasets(
    strategy,
    dataset_paths,
    dataset_labels,
    save_path=None,
    measurement_uncertainty=0.005,
    title=None
):
    # Manually define distinct colors (vibrant and high contrast)
    distinct_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))  # or (5.5, 3) for compact paper format
    iterations = list(range(100))

    # Plot each dataset
    for i, path in enumerate(dataset_paths):
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue

        df = pd.read_csv(path)
        if strategy not in df.columns:
            print(f"[Warning] Strategy '{strategy}' not found in: {path}")
            continue

        raw_values = df[strategy]
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values[:100]

        color = distinct_colors[i % len(distinct_colors)]
        label = dataset_labels[i] if i < len(dataset_labels) else f"Dataset {i+1}"

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle='-',       # Solid line
            linewidth=1.5        # Thin line (not bold)
        )

    # Measurement Uncertainty Line
    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Measurement Uncertainty'
    )

    # Formatting
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 100])

    if title:
        ax.set_title(title, fontsize=14, pad=12)

    ax.legend(
        title="Dataset",
        fontsize=10,
        title_fontsize=12,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.3,
        columnspacing=1.0
    )

    plt.tight_layout()

    if save_path:
        if not save_path.lower().endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
       
    else:
        plt.show()


def get_large_color_palette(n):
    """Return n distinct colors by combining colormaps."""
    cmap_list = ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set3', 'Paired', 'Pastel1']
    color_list = []

    for cmap_name in cmap_list:
        cmap = cm.get_cmap(cmap_name)
        for i in range(cmap.N):
            rgba = cmap(i)
            color_list.append(mcolors.to_hex(rgba))
            if len(color_list) >= n:
                return color_list
    return color_list[:n]

def plot_all_base_and_mixed_strategies(df, main_strategy, base_strategies, save_path=None, measurement_uncertainty=0.005):
    strategies_to_plot = []
    labels = []
    styles = []

    full_strategy_list = []

    for base in base_strategies:
        full_strategy_list.append(base)
        full_strategy_list.append(f"{main_strategy}+{base}")

    color_palette = get_large_color_palette(len(full_strategy_list))
    color_map = dict(zip(full_strategy_list, color_palette))

    for base in base_strategies:
        base_label = STRATEGY_DISPLAY_NAMES.get(base, base)
        mixed_label = f"{STRATEGY_DISPLAY_NAMES.get(main_strategy, main_strategy)}+{base_label}"

        if base in df.columns:
            strategies_to_plot.append(base)
            labels.append(base_label)
            styles.append(("solid", color_map[base]))

        mixed_name = f"{main_strategy}+{base}"
        if mixed_name in df.columns:
            strategies_to_plot.append(mixed_name)
            labels.append(mixed_label)
            styles.append(("dashed", color_map[mixed_name]))

    if not strategies_to_plot:
        print("No matching strategies found in the data.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = list(range(100))

    for strategy, label, (linestyle, color) in zip(strategies_to_plot, labels, styles):
        raw_values = df[strategy] if strategy in df.columns else pd.Series([None]*100)
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values[:100]

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )

    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Measurement Uncertainty'
    )

    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 100])

    ax.legend(
        title="Strategy",
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.4,
        columnspacing=1.2
    )

    plt.tight_layout()

    if save_path:
        if not save_path.endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()




def plot_initialization_strategies(csv_path, all_init_strategies,
                                   resistance_col="Resistance", x_col="x", y_col="y",
                                   max_points=342, output_path=None):
    """
    Plot initialization strategies on a wafer-like grid with color-coded resistance values.
    Selected initialization points are highlighted for each strategy.

    Args:
        csv_path (str): Path to the dataset CSV file.
        all_init_strategies (dict): Dictionary {strategy_name: list of selected indices}.
        resistance_col (str): Name of the resistance column in the CSV.
        x_col (str): Column name for x-coordinates.
        y_col (str): Column name for y-coordinates.
        max_points (int): Maximum number of rows to consider from the dataset.
        output_path (str, optional): If provided, saves the plot to this PDF path.
    """
    data = pd.read_csv(csv_path).iloc[:max_points]
    x = data[x_col].values
    y = data[y_col].values
    resistance = data[resistance_col].values

    cmap = plt.colormaps["plasma"]
    num_strategies = len(all_init_strategies)
    cols = 4
    rows = (num_strategies // cols) + (num_strategies % cols > 0)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for idx, (strategy, indices) in enumerate(all_init_strategies.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        ax.set_aspect("equal")

        ax.scatter(x, y, c=resistance, cmap=cmap, marker="s", s=50)
        ax.scatter(x[indices], y[indices], c="white", marker="X", s=200,
                   edgecolor="black", linewidth=2)
        ax.scatter(x[indices], y[indices], c="red", marker="o", s=100,
                   edgecolor="black", linewidth=1, alpha=0.8)

        for i in indices:
            ax.text(x[i], y[i], 'X', fontsize=12, color='black',
                    ha='center', va='center', fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(strategy, fontsize=16)

        if idx == 0:
            ax.legend(["Init Points"], loc="upper right", fontsize=8)

    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    # Save if output_path is provided
    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches='tight', dpi=300)
      
    else:
        plt.show()