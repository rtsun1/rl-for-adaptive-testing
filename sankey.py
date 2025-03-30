import plotly.graph_objects as go
import pandas as pd
import numpy as np

def sankey(datasets):
    all_sequences = []
    for dataset in datasets:
        all_sequences.extend(dataset["histories"].item())
    fig = sankey_from_action_sequences(all_sequences,min_frequency=1, max_depth=5)
    fig.show()
    
def sankey_from_action_sequences(sequences, min_frequency=1, max_depth=None):
    """
    Generates a Sankey diagram from a list of action sequences, with depth control.

    Args:
        sequences: A list of lists, where each inner list represents a sequence
                   of tions (represented by IDs, strings, or any hashable type).
        min_frequency: Minimum number of times a link must appear to be included.
        max_depth: The maximum depth (number of steps from the start) to visualize.
                   If None, all steps are included.

    Returns:
        A plotly.graph_objects.Figure object representing the Sankey diagram.
    """

    links = []
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            if max_depth is None or i < max_depth:
                source_label = f"Q{sequence[i]}-#{i}"  # Add depth to source
                target_label = f"Q{sequence[i+1]}-#{i+1}" # Add depth to target
                links.append((source_label, target_label))


    df = pd.DataFrame(links, columns=['source', 'target'])
    link_counts = df.groupby(['source', 'target']).size().reset_index(name='value')
    link_counts = link_counts[link_counts['value'] >= min_frequency]

    all_nodes = list(link_counts['source'].unique()) + list(link_counts['target'].unique())
    all_nodes = sorted(list(set(all_nodes)))
    node_labels = [str(node) for node in all_nodes]
    label_to_index = {label: i for i, label in enumerate(node_labels)}

    link_counts['source_index'] = link_counts['source'].map(label_to_index)
    link_counts['target_index'] = link_counts['target'].map(label_to_index)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
        ),
        link=dict(
            source=link_counts['source_index'],
            target=link_counts['target_index'],
            value=link_counts['value'],
        )
    )])

    fig.update_layout(title_text="Action Sequence Sankey Diagram", font_size=12)
    return fig

from pprint import pprint
datasets = [np.load("runs/CAT-v0__dqn__42__1742417085/results.npz", allow_pickle=True)]

data = datasets[0]["histories"].item()
counts = {}
for i in range(len(data)):
    counts[i] = counts.get(data[i][0], 0) + 1
# pprint(counts)

sankey(datasets)