from __future__ import annotations

from utilities import convert_var_list_rev
from data_handling.coordinates import CoordinateFrame

############################################ USER SETTINGS #############################################################
pinn_cl_metrics: dict[str,dict[str,int|float|None]] = {
    "pinn_metrics": {
        "num_layers": 1, "num_neurons": 5
    },
    "adam_metrics": {
        "iterations": 200, "target_val": None, "val_interval": None, "learning_rate": 1e-3, "batch_size": 1_400
    },
    "lbfgs_metrics": {
        "iterations": 50, "target_val": None, "val_interval": None
    }
}

pinn_gc_metrics: dict[str,dict[str,int|float|None]] = {
    "pinn_metrics": {
        "num_layers": 10, "num_neurons": 15
    },
    "adam_metrics": {
        "iterations": 12_000, "target_val": None, "val_interval": None, "learning_rate": 1e-3, "batch_size": 1_000
    },
    "lbfgs_metrics": {
        "iterations": 10_000, "target_val": None, "val_interval": None
    }
}

ann_cl_metrics: dict[str,dict[str,int|float|None]] = {
    "pinn_metrics": {
        "num_layers": 1, "num_neurons": 5
    },
    "adam_metrics": {
        "iterations": 200, "target_val": None, "val_interval": None, "learning_rate": 1e-3, "batch_size": 1_400
    },
    "lbfgs_metrics": {
        "iterations": 50, "target_val": None, "val_interval": None
    }
}

ann_gc_metrics: dict[str,dict[str,int|float|None]] = {
    "pinn_metrics": {
        "num_layers": 10, "num_neurons": 15
    },
    "adam_metrics": {
        "iterations": 12_000, "target_val": None, "val_interval": None, "learning_rate": 1e-3, "batch_size": 1_000
    },
    "lbfgs_metrics": {
        "iterations": 10_000, "target_val": None, "val_interval": None
    }
}
############################################ USER SETTINGS #############################################################

def set_metrics(network: str) -> dict[str, dict[str, int | float | None]]:
    metrics: dict[str, dict[str, int | float | None]] = {
        # This will change the data metrics for all networks.
        "data_metrics": {"vel_lim": 0.1, "tol": 0.2, "slice_pos": None}
    }

    if network == "PINN-CL":
        metrics.update(pinn_cl_metrics)
    elif network == "PINN-GC":
        metrics.update(pinn_gc_metrics)
    elif network == "ANN-CL":
        metrics.update(ann_cl_metrics)
    elif network == "ANN-GC":
        metrics.update(ann_gc_metrics)
    else:
        raise ValueError(
            f"'network = {network}' doesn't comply with one of the options "
            f"'PINN-CL', 'PINN-GC', 'ANN-CL', or 'ANN-GC'."
        )
    return metrics

def set_coordinates(network: str) -> CoordinateFrame:
    if "CL" in network:
        return CoordinateFrame("Z (m)","Y (m)","X (m)")

    if "GC" in network:
        return CoordinateFrame("Z (m)","X (m)","Y (m)")

    raise ValueError(
        f"'network = {network}' doesn't have either 'CL' or 'GC' as an object of investigation."
    )

def set_outputs(network: str) -> list[str]:
    if "CL" in network:
        return convert_var_list_rev(
            ["p", "u", "v", "w", "Y_O2", "j", "c_O2", "eta", "U_eq", "Y_H2", "Y_H2O", "Y_N2"]
        )

    if "GC" in network:
        return convert_var_list_rev(["p", "u", "v", "w", "Y_O2"])

    raise ValueError(
        f"'network = {network}' doesn't have either 'CL' or 'GC' as an object of investigation."
    )

def set_equations(network: str) -> dict[str, bool]:
    if "ANN" in network:
        return {"conti": False, "momentum": False, "spec_trans": False, "react_rate": False}

    elif "PINN" in network:
        if "CL" in network:
            return {"conti": False, "momentum": False, "spec_trans": False, "react_rate": True}

        if "GC" in network:
            return {"conti": True, "momentum": True, "spec_trans": True, "react_rate": False}

        raise ValueError(
            f"'network = {network}' doesn't have either 'CL' or 'GC' as an object of investigation."
        )

    raise ValueError(
        f"'network = {network}' doesn't have either 'ANN' or 'PINN' as an object of investigation."
    )