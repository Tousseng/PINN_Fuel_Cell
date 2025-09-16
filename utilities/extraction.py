import os

def extract_info(file_path: str) -> tuple[float,str,str]:
    file_name: str = os.path.basename(file_path).split(".")[0]
    first, sec = file_name.split("mm")[0:2]
    length_mm: float = float("".join(list(first)[-3:]).replace("_","."))
    part: str = "".join(list(first)[:-3])
    segment_no: str = sec
    return length_mm * 10 ** (-3), part, segment_no