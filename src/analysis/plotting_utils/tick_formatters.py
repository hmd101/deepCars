import matplotlib
from matplotlib.ticker import FuncFormatter


def sizeof_fmt(x, pos):
    if x < 0:
        return ""
    for x_unit in ["bytes", "kB", "MB", "GB", "TB"]:
        if x < 1000.0:
            return "%3.0f %s" % (x, x_unit)
        x /= 1000.0


def timeof_fmt(x, pos):
    if x < 0.0:
        return ""
    if x < 60.0:
        return f"{x:2.0f} s"
    x /= 60.0
    if x < 60:
        return f"{x:2.0f} min"
    x /= 60.0
    if x < 24:
        return f"{x:2.0f} h"
    x /= 24.0
    if x < 365:
        return f"{x:3.0f} d"
    x /= 365
    return f"{x:3.0f} y"


MemoryFormatter = lambda: FuncFormatter(sizeof_fmt)
TimeFormatter = lambda: FuncFormatter(timeof_fmt)
