import re
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def extract_method(path: str) -> str:
    if "PlusPlus" in path:
        return "gradcamPlusPlus"
    elif "Layer" in path:
        return "gradcamLayer"
    else:
        return "gradcam"


def extract_layer(path: str) -> int:
    match = re.search(r"_L(\d+)", path)
    return int(match.group(1)) if match else None


def extract_magnification(path: str) -> str:
    match = re.search(r"_(\d+)x", path)
    return match.group(1) + "x" if match else ""


def parse_log(file_path):
    pattern = re.compile(
        r"Heatmap1: (?P<h1>\S+) Heatmap2: (?P<h2>\S+)\. SSIM: (?P<ssim>[\d.]+) Cosine Similarity: (?P<cos>[\d.]+) Mean Squared Error: (?P<mse>[\d.]+)"
    )
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            h1 = m.group('h1')
            h2 = m.group('h2')
            ssim = float(m.group('ssim'))
            cos = float(m.group('cos'))
            mse = float(m.group('mse'))

            row = {
                'Method1': extract_method(h1),
                'Method2': extract_method(h2),
                'Layer': extract_layer(h1),
                'Size1': extract_magnification(h1),
                'Size2': extract_magnification(h2),
                'SSIM': ssim,
                'Cosine': cos,
                'MSE': mse
            }
            rows.append(row)
    return pd.DataFrame(rows)


# --- New helper function for plotting ---
def create_and_save_heatmap(data, title, filename, vmin=0, vmax=1):
    """Creates, styles, and saves a seaborn heatmap."""
    if data.empty:
        print(f"Warning: No data available for '{title}'. Skipping plot.")
        return

    # Increase font size for axis titles (e.g., "Layer", "Method")
    # --- END OF MODIFICATIONS ---
    plt.figure(figsize=(8, 6))
    plt.xlabel(data.columns.name, fontsize=14)
    plt.ylabel(data.index.name, fontsize=14)

    # Increase font size for tick labels (e.g., "24", "26", "GradCAM")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)  # rotation=0 keeps y-labels horizontal
    sns.heatmap(
        data,
        annot=True,  # Show the SSIM values on the map
        fmt=".2f",  # Format values to 2 decimal places
        cmap="viridis",  # Color scheme
        linewidths=.5,
        vmin=vmin,  # Set min for color bar consistency
        vmax=vmax  # Set max for color bar consistency
    )
    plt.title(title, fontsize=16)
    plt.xlabel(data.columns.name, fontsize=12)
    plt.ylabel(data.index.name, fontsize=12)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    print(f"Saved: {filename}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python comparison_table.py <path_to_log_file>")
        sys.exit(1)

    log_path = sys.argv[1]
    df = parse_log(log_path)

    # Table 1: Comparing Sizes
    df_sizes = df[
        (df["Method1"] == df["Method2"]) &
        (df["Size1"] != df["Size2"])
        ]
    # Rename Method1 to Method for clarity in plots
    df_sizes = df_sizes.rename(columns={'Method1': 'Method'})
    df_sizes_mean = (
        df_sizes.groupby(['Method', 'Layer', 'Size1', 'Size2'], as_index=False)
        [['SSIM', 'Cosine', 'MSE']].mean().round(2)
    )

    # Table 2: Comparing Methods
    df_methods = df[
        (df["Method1"] != df["Method2"]) &
        (df["Size1"] == df["Size2"])
        ]
    df_methods_mean = (
        df_methods.groupby(['Method1', 'Method2', 'Layer', 'Size1'], as_index=False)
        [['SSIM', 'Cosine', 'MSE']].mean().round(2)
    )

    # Output results (as before)
    print("\n=== Średnie wartości — Porównanie rozdzielczości (Sizes) ===")
    print(df_sizes_mean.to_string(index=False))

    print("\n=== Średnie wartości — Porównanie metod (Methods) ===")
    print(df_methods_mean.to_string(index=False))

    # --- New section for generating and saving heatmaps ---
    print("\n=== Generating SSIM Heatmaps ===")

    # --- Heatmaps for Method Comparisons ---
    method_pairs = [
        ("gradcam", "gradcamPlusPlus"),
        ("gradcam", "gradcamLayer"),
        ("gradcamPlusPlus", "gradcamLayer")
    ]

    for m1, m2 in method_pairs:
        # Filter for the specific pair of methods
        filtered_df = df_methods_mean[
            (df_methods_mean['Method1'] == m1) &
            (df_methods_mean['Method2'] == m2)
            ]

        method_name_map = {
            'gradcam': 'GradCAM',
            'gradcamPlusPlus': 'GradCAM++',
            'gradcamLayer': 'LayerCam'
        }
        # Pivot to create a 2D matrix: Layers on X-axis, Magnification on Y-axis
        pivot_df = filtered_df.pivot(index='Size1', columns='Layer', values='SSIM')
        m1 = method_name_map[m1]
        m2 = method_name_map[m2]
        # Define title and filename
        title = f'SSIM Heatmap ({m1} vs {m2})'
        filename = f'heatmap_ssim_{m1}_vs_{m2}.png'

        create_and_save_heatmap(pivot_df, title, filename)

    # --- Heatmaps for Magnification Comparisons ---
    magnification_pairs = [
        ("10x", "20x"),
        ("20x", "40x"),
        ("10x", "40x")  # Corrected from 40x vs 10x for consistency
    ]

    # Map for prettier Y-axis labels
    method_name_map = {
        'gradcam': 'GradCAM',
        'gradcamPlusPlus': 'GradCAM++',
        'gradcamLayer': 'LayerCam'
    }

    for s1, s2 in magnification_pairs:
        # Filter for the specific pair of magnifications
        # We check both (s1, s2) and (s2, s1) to be robust
        filtered_df = df_sizes_mean[
            ((df_sizes_mean['Size1'] == s1) & (df_sizes_mean['Size2'] == s2)) |
            ((df_sizes_mean['Size1'] == s2) & (df_sizes_mean['Size2'] == s1))
            ]

        # Pivot to create a 2D matrix: Layers on X-axis, Method on Y-axis
        pivot_df = filtered_df.pivot(index='Method', columns='Layer', values='SSIM')

        # Use nicer names for the Y-axis
        pivot_df = pivot_df.rename(index=method_name_map)
        desired_order = ['GradCAM', 'GradCAM++', 'LayerCam']
        pivot_df = pivot_df.reindex(desired_order)
        # Define title and filename
        title = f'SSIM Heatmap ({s1} vs {s2})'
        filename = f'heatmap_ssim_{s1}_vs_{s2}.png'

        create_and_save_heatmap(pivot_df, title, filename)

    print("\n=== All heatmaps saved successfully! ===")


if __name__ == "__main__":
    main()
