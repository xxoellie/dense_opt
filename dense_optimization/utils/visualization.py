import tensorflow as tf
from pathlib import Path


def visualize_model(model: tf.keras.Model, prefix: str, output_dir: str):
    output_dir = Path(output_dir)
    file_path = output_dir.joinpath(f"{prefix}_{model.name}_vis")
    dot = tf.keras.utils.model_to_dot(
        model=model,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96
    )
    for i, node in enumerate(dot.get_node_list()):
        if "label" not in node.get_attributes():
            continue
        for la in {tf.keras.layers.BatchNormalization}:
            if f": {la.__name__}\n" in node.get_attributes()["label"]:
                node.add_style("filled")
                break
        node.get_attributes()["label"] = f"{i}. {node.get_attributes()['label']}"
    # dot.write(path=str(file_path)+".svg", format="svg")
    dot.write(path=str(file_path)+".png", format="png")
