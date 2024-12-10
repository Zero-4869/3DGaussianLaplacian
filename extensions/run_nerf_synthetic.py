# training scripts for the nerf-synthetic datasets
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools

# scenes = ["cat4", "centaur2", "centaur3", "centaur4", "david6", "dog0", "dog1", "dog2", "dog3", "dog5","horse6", "horse7", "horse10",\
# "michael0", "michael1", "michael2", "michael3", "michael4", "victoria2", "victoria4", "victoria7", "wolf1", "wolf2" ]
# scenes = ["cat2", "cat3", "cat4", "centaur0", "centaur1", "centaur2", "centaur3", "centaur4", \
#           "david0", "david1", "david6", "david10", "david11", "dog0", "dog1", "dog2", "dog3", "dog5", \
#             "gorilla1", "gorilla5", "gorilla8", "gorilla14", "horse0", "horse5", "horse6", "horse7", "horse10", \
#             "michael0", "michael1", "michael2", "michael3", "michael4", "victoria0", "victoria1", "victoria2", "victoria4", "victoria7", 
#             "wolf0", "wolf1", "wolf2"]
scenes = ["david10", "david11"]
factors = [1]

output_dir = "exp_nerf_synthetic/adaptive_training"

dataset_dir = "data/nerf_synthetic"

dry_run = False

excluded_gpus = set([])

# 1: 3000
# 2: 1000 10000 16000, densify:15000
# 3: densify:7000, pruning: 8000 # not good
# 4: densify:7000, pruning: 9000 13000 # not good
# 5: densify:15000, pruning: iteration # sparcity

jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):
    iterations = " ".join([str(x) for x in range(0, 32000, 2000)])
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train_laplacian.py -s {dataset_dir}/{scene} -m {output_dir}/{scene}_f_pc5 --eval --white_background --port {6209+int(gpu)} \
        --save_iterations {iterations}\
            --test_iterations {iterations}\
            --iterations 30000\
            --prune_iterations_cov {iterations}\
            --densify_until_iter 15000\
                --position_lr_max_steps 30000"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    return True

    
def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

