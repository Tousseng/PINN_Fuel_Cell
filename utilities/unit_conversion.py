import numpy as np

conversion: dict[str,tuple[str,float]] = {r"m": (r"mm", 1e3),
                                          r"A/m$^2$": (r"A/cm$^2$", 1e-4)
                                          }

def convert(unit: str, data: np.ndarray) -> tuple[str,np.ndarray]:
    factor: float = 1.0
    if unit in list(conversion.keys()):
        unit, factor = conversion[unit]

    return unit, factor * data