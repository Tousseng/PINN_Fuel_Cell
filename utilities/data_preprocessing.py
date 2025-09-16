import tensorflow as tf

def split_collocation_and_boundary(
        inputs: tf.Tensor, outputs: tf.Tensor, in_indices: dict[str, int], epsilon: float) -> tuple[tf.Tensor, ...]:
    """
    Split the supplied positional data into collocation and boundary points and return them.
    Args:
        inputs: Tensorflow data of the position x, y, and z.
        outputs: Tensorflow data of the outputs.
        in_indices: Indices to map the inputs x, y, and z to their input position to the neural network.
        epsilon: Percentage of the distance between the data points and their respective minimum
                 and maximum relative to the difference of maximum and minimum to be included.

    Returns:
        Tensorflow tensors of the boundary inputs, boundary outputs, collocation inputs, and collocation outputs
        as well as the unchanged inputs and outputs.
    """
    pos_boundaries: dict[str, dict[str, tf.Tensor]] = {}
    for coord, coord_idx in in_indices.items():
        if coord_idx == -1:
            continue
        pos_data: tf.Tensor = inputs[:, coord_idx]
        pos_min: tf.Tensor = tf.reduce_min(pos_data)
        pos_max: tf.Tensor = tf.reduce_max(pos_data)
        distance_to_min: tf.Tensor = (pos_data - pos_min) / (pos_max - pos_min) * tf.constant(100, dtype=tf.float32)
        distance_to_max: tf.Tensor = (pos_max - pos_data) / (pos_max - pos_min) * tf.constant(100, dtype=tf.float32)
        min_mask: tf.Tensor = tf.less_equal(distance_to_min, epsilon)
        max_mask: tf.Tensor = tf.less_equal(distance_to_max, epsilon)
        boundary_vals_min: tf.Tensor = tf.squeeze(tf.where(min_mask))
        boundary_vals_max: tf.Tensor = tf.squeeze(tf.where(max_mask))
        pos_boundaries[coord] = {"lb": boundary_vals_min, "ub": boundary_vals_max}

    # <tf.unique(...).y> returns each index only once at its first appearance.
    boundary_indices: tf.Tensor = tf.unique(
        tf.concat([tf.concat([d["lb"], d["ub"]], axis=0) for d in list(pos_boundaries.values())], axis=0)).y

    all_indices = tf.range(inputs.shape[0], dtype=tf.int64)
    mask_not_boundary = ~tf.reduce_any(
        tf.equal(all_indices[:, tf.newaxis], boundary_indices), axis=1
    )
    collocation_indices = tf.boolean_mask(all_indices, mask_not_boundary)

    boundary_inputs: tf.Tensor = tf.gather(inputs, boundary_indices)
    boundary_outputs: tf.Tensor = tf.gather(outputs, boundary_indices)
    collocation_inputs: tf.Tensor = tf.gather(inputs, collocation_indices)
    collocation_outputs: tf.Tensor = tf.gather(outputs, collocation_indices)

    return boundary_inputs, boundary_outputs, collocation_inputs, collocation_outputs, inputs, outputs