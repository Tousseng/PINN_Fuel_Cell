linestyles = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot1": "-.",
    "dashdot2": (0, (3, 1, 1, 1)),
    "dashdot3": (0, (5, 2, 2, 2)),
    "dashdot4": (0, (4, 2, 1, 2)),
    "dashdot5": (0, (6, 2, 1, 2)),
    "dashdot6": (0, (4, 1, 1, 1, 1, 1)),
}

def linestyle_list_for_plot(line_list: list[str]) -> list[str]:
    linestyle_list: list[str] = []
    for line in line_list:
        linestyle_list.append(linestyles[line])

    return linestyle_list

if __name__ == "__main__":
    l = ["solid", "dashed", "dashdot2"]
    print(linestyle_list_for_plot(l))