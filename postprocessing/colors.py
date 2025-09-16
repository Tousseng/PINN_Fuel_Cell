from typing import Union

# Colors defined by RGB values
colors = {
    "White": [255,255,255],
    "White1": [242,242,242],
    "White2": [216,216,216],
    "White3": [191,191,191],
    "White4": [165,165,165],
    "White5": [127,127,127],
    "Black": [0, 0, 0],
    "Black1": [127,127,127],
    "Black2": [89,89,89],
    "Black3": [63,63,63],
    "Black4": [38,38,38],
    "Black5": [12,12,12],
    "Grey": [121,121,121],
    "Grey1": [228,228,228],
    "Grey2": [201,201,201],
    "Grey3": [174,174,174],
    "Grey4": [90,90,90],
    "Grey5": [60,60,60],
    "DarkBlue": [0, 84, 159],
    "DarkBlue1": [184,221,255],
    "DarkBlue2": [114,188,255],
    "DarkBlue3": [44,155,255],
    "DarkBlue4": [0,62,119],
    "DarkBlue5": [0,41,79],
    "LightBlue": [142,186,229],
    "LightBlue1": [232,241,249],
    "LightBlue2": [209,227,244],
    "LightBlue3": [187,213,239],
    "LightBlue4": [66,139,211],
    "LightBlue5": [34,93,150],
    "DarkGrey": [60,60,60],
    "DarkGrey1": [216,216,216],
    "DarkGrey2": [177,177,177],
    "DarkGrey3": [137,137,137],
    "DarkGrey4": [45,45,45],
    "DarkGrey5": [30,30,30],
    "Green": [122,181,29],
    "Green1": [229,246,203],
    "Green2": [204,238,151],
    "Green3": [179,230,99],
    "Green4": [91,135,21],
    "Green5": [60,90,14],
    "DarkBlueGreen": [48,102,109],
    "DarkBlueGreen1": [205,230,233],
    "DarkBlueGreen2": [156,205,211],
    "DarkBlueGreen3": [107,180,190],
    "DarkBlueGreen4": [36,76,81],
    "DarkBlueGreen5": [24,51,54],
    "LightGrey": [217,217,217],
    "LightGrey1": [195,195,195],
    "LightGrey2": [162,162,162],
    "LightGrey3": [108,108,108],
    "LightGrey4": [54,54,54,255],
    "LightGrey5": [21,21,21],
    "Orange": [242,148,0],
    "Orange1": [255,234,201],
    "Orange2": [255,213,147],
    "Orange3": [255,192,94],
    "Orange4": [181,111,0],
    "Orange5": [121,74,0],
    "Red": [255,0,0],
    "DarkRed": [110, 16, 32],
    "LightRed": [251, 225, 229]
}

def color_list_for_plot(colors_list: list[str]) -> list[tuple[Union[float, any], ...]]:
    rgb_list: list[tuple[Union[float,any], ...]] = []
    for color in colors_list:
        # Transformation of each RGB value from [0, 255] to [0, 1] (for matplotlib)
        rgb_list.append(tuple([val / 255 for val in colors[color]]))
    return rgb_list

def color_for_plot(color: str) -> tuple[float, ...]:
    return tuple([val / 255 for val in colors[color]])

if __name__ == "__main__":
    c = ["Red", "darkBlue1"]
    print(color_list_for_plot(c))