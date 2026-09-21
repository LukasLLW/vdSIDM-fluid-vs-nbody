from pathlib import Path
import os
current_path = os.path.abspath(__file__).replace("\\", "/")
code_index = current_path.rfind("/code/")
MAIN_DIR = Path(current_path[:code_index])
def Nbody_root(rest):
    return MAIN_DIR / "database" /fr"Fischer_Data"/rest

def gravothermal_root(rest):
    return  MAIN_DIR / "database" / rest

def data_root(rest):
    return  MAIN_DIR / "database" / rest

def figure_Path(rest):
    return MAIN_DIR / "figures" / rest


