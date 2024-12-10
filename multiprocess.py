import os
from multiprocessing import Pool

def run_script(scripts):
    with Pool(processes=10) as pool:
        pool.map(os.system, scripts)
