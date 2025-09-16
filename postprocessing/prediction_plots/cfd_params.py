active_area: float = 3.111381 # cm^2
norm_area: float = 1.5778 # cm^2

channel_widths: list[float] = [0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0010, 0.0011]
curr_dens_cathode_list: list[float] = [2.518353, 2.132814, 1.872399, 1.684270, 1.519585, 1.311553, 1.183027] # A/cm^2

# area (cm^2) of the cathode voltage collector
area_cathode_list: list[float] = [0.6515366, 0.8777366, 1.1039370, 1.3301370, 1.5563370, 2.0087370, 2.2349370]

# normalized by multiplying the cathode current density with the true area of the cathode voltage collector
#       and dividing by the area of the anode voltage collector (norm_area)
curr_dens_cathode_norm_list: list[float] = [1.039928, 1.186493, 1.310058, 1.419894, 1.498913, 1.669771, 1.675744]

def transform_react_rate_to_curr_dens(react_rate: float) -> float:
    #idx: int = channel_widths.index(width)
    return react_rate * active_area / norm_area * 1e-4

if __name__ == "__main__":
    c =transform_react_rate_to_curr_dens(5372.369756099077)
    print(c)