import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

def plot_manhattan(data, category_col, group_col, y_col, area_col=None, transform=None, figsize=(18, 12), point_size=50, wrap_width=35):
    plot_data = data.copy()
    plot_data['y_transformed'] = plot_data[y_col].apply(transform) if transform else plot_data[y_col]
    plot_data['category'] = plot_data[category_col].astype(str)
    plot_data['group'] = plot_data[group_col].astype(str)
    plot_data = plot_data.sort_values(by=[group_col, category_col])
    plot_data['x'] = range(len(plot_data))

    # Scale area_col for point size if provided
    sizes = point_size if area_col is None else plot_data[area_col] / plot_data[area_col].max() * (point_size * 10)

    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=plot_data,
        x='x',
        y='y_transformed',
        hue='group',          # Use hue for group categories
        palette="tab10",
        size=sizes,           # Apply sizes for scaling points
        sizes=(point_size, point_size * 10)  # Define range for sizes
    )

    # Extract handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    # Keep only the labels related to the hue (categories/groups)
    ax.legend(handles[:len(plot_data['group'].unique()) + 1],  # Adjust index for hue labels
              labels[:len(plot_data['group'].unique()) + 1],
              bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=13)

    # Calculate group tick positions
    group_ticks = []
    for grp in plot_data['group'].unique():
        grp_data = plot_data[plot_data['group'] == grp]
        mid_point = grp_data['x'].iloc[0] + (grp_data['x'].iloc[-1] - grp_data['x'].iloc[0]) / 2
        group_ticks.append(mid_point)

    # Wrap x-axis tick labels to avoid stretching
    wrapped_labels = ['\n'.join(textwrap.wrap(label, wrap_width)) for label in plot_data['group'].unique()]
    plt.xticks(ticks=group_ticks, labels=wrapped_labels, rotation=45, ha='right')

    plt.xlabel("Group")
    plt.ylabel(f"{y_col} (transformed)" if transform else y_col)
    plt.title("Manhattan Plot")
    plt.tight_layout()
