from preprocessing.data_analysis.utils import *

def sort_data_points(bound: tf.Tensor, coll: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x_bound: np.ndarray = bound.numpy()[:, 0]
    y_bound: np.ndarray = bound.numpy()[:, 1]

    x_coll: np.ndarray = coll.numpy()[:, 0]
    y_coll: np.ndarray = coll.numpy()[:, 1]

    sort_index_bound: np.ndarray = np.lexsort((x_bound, y_bound))
    sort_index_coll: np.ndarray = np.lexsort((x_coll, y_coll))

    return tf.gather(bound, sort_index_bound), tf.gather(coll, sort_index_coll)

def reduce_data_points(sorted_tensors: list[tf.Tensor], n: int) -> tuple[tf.Tensor, tf.Tensor]:
    results: list[tf.Tensor] = []
    for sorted_in in sorted_tensors:
        y_vals: np.ndarray = sorted_in.numpy()[:, 1]
        unique_y: np.ndarray = np.unique(y_vals)

        selected_points = []

        # Loop over each unique y-group
        for y in unique_y:
            group: np.ndarray = sorted_in.numpy()[y_vals == y]

            # x-values only
            x_vals = group[:, 0]

            # Indices: first, last, and the every n-th in between
            indices = [0]  # first
            if len(x_vals) > 1:
                indices += list(range(15, len(x_vals) - 10, n))  # every n-th
                indices += [len(x_vals) - 1]  # last

            # Extract the data points (x and y)
            selected_points += list(group[indices])

        results.append(tf.constant(selected_points))

    return results[0], results[1]

if __name__ == "__main__":
    data: pd.DataFrame = load_data().reset_index()

    path: str = filedialog.askdirectory(title="Choose a directory to save the plot in.")
    plot_name: str = "Z-X-Positions_Split.pdf"

    x_str, y_str = "z", "x"

    arr_x: np.ndarray = select_data(data, x_str)
    arr_y: np.ndarray = select_data(data, y_str)

    x = tf.convert_to_tensor(select_data(data, "x"), dtype=tf.float32)
    z = tf.convert_to_tensor(select_data(data, "z"), dtype=tf.float32)
    positions = tf.stack([z, x], axis=1)
    w = tf.convert_to_tensor(select_data(data, "w"), dtype=tf.float32)

    bound_in, bound_out, coll_in, coll_out, _, _ = split_collocation_and_boundary(
        inputs=positions, outputs= w, in_indices={"z": 0, "x": 1}, epsilon=1
    )

    sorted_bound_in, sorted_coll_in = sort_data_points(bound=bound_in, coll=coll_in)

    sorted_bound_in, sorted_coll_in = reduce_data_points(
        sorted_tensors=[sorted_bound_in, sorted_coll_in],
        n=15
    )

    fig = plot_split_inputs(
        boundary_inputs=sorted_bound_in.numpy(),
        collocation_inputs=sorted_coll_in.numpy(),
        x_str=x_str, y_str=y_str
    )

    save_plot(fig=fig, path=path, filename=plot_name)