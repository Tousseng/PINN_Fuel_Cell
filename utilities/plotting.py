from matplotlib.pyplot import Axes

def annotate_interval(ax: Axes, x_start: float, x_end: float, y_pos: float, text: str):
    ax.annotate('', xy=(x_end, y_pos), xytext=(x_start, y_pos),
                 arrowprops=dict(arrowstyle="<->", color="grey")
                )

    ax.text((x_start + x_end) / 2, y_pos, text,
             ha="center", va="center", color="grey",
             bbox=dict(facecolor="white", alpha=1.0, edgecolor="white", boxstyle="square,pad=0.15")
             )