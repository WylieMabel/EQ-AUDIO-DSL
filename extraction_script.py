import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import os
from sklearn.model_selection import train_test_split 
import pandas as pd

# --- Configuration (Adjust these paths) ---
HDF5_FILE_PATH = '/Users/mabelwylie/Documents/EQ-AUDIO-DSL/raw_data/merge.hdf5' # Replace with the actual path to your HDF5 file
OUTPUT_DIR = 'extracted_data_optimized'
EV_LIST = [f'event_{i}' for i in range(1000)] # Placeholder: Replace with your actual ev_list
N_PROCESSES = 8 # Use 4 cores as requested
TEST_SIZE = 0.10 # 10% for test, 90% for train
RANDOM_STATE = 42 # Set for reproducible splits

# --- Core Worker Logic (Reads Input HDF5) ---
def process_event_core(evi, idx, input_hdf5_path):
    """
    Worker function: reads data, separates channels, and returns the small arrays.
    Crucially, NO DISK I/O happens in this function to prevent locking issues.
    """
    try:
        # 1. READ DATA (from input HDF5)
        # Each worker opens the input HDF5 file for read-only access
        with h5py.File(input_hdf5_path, 'r') as dtfl_in:
            dataset = dtfl_in.get('data/' + str(evi))
            data = np.array(dataset)
        
        # Separate the three channels
        E_data = data[:, 0]
        N_data = data[:, 1]
        Z_data = data[:, 2]
        
        # 2. RETURN DATA (for main process to write to disk)
        # Returns the index, event ID, and the three NumPy arrays.
        return idx, evi, E_data, N_data, Z_data
        
    except Exception as e:
        # Use print() for error reporting since workers don't have good communication channels
        print(f"Error processing event {evi} at index {idx}: {e}")
        return idx, evi, None, None, None # Return None for data on failure

# --- Worker Function for imap_unordered ---
def imap_event_worker(task_tuple, input_hdf5_path):
    """Unpacks the task tuple (evi, idx) and calls the core processing function."""
    evi, idx = task_tuple
    return process_event_core(evi, idx, input_hdf5_path)

# --- Pipeline Manager Function ---
def run_extraction_pipeline(ev_list, input_hdf5_path, output_dir, n_processes, dataset_name):
    """
    Manages the multiprocessing pool, streams results from workers, and writes 
    them immediately to the three separate output HDF5 files.
    """
    print(f"Starting parallel extraction for {dataset_name} using {n_processes} cores...")
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []
    
    # Define the three distinct output file paths
    E_path = os.path.join(output_dir, f'{dataset_name}_E_data.h5')
    N_path = os.path.join(output_dir, f'{dataset_name}_N_data.h5')
    Z_path = os.path.join(output_dir, f'{dataset_name}_Z_data.h5')
    
    # Pre-delete the output files to ensure a clean start
    for p in [E_path, N_path, Z_path]:
        if os.path.exists(p):
            os.remove(p)

    # Setup the partial function
    worker_func = partial(imap_event_worker, input_hdf5_path=input_hdf5_path)
    
    # Prepare the iterable for the pool
    task_iterable = [(evi, i) for i, evi in enumerate(ev_list)]

    # --- Initialize HDF5 Datasets in Main Process ---
    # We must open all three files here and close them at the end.
    h5_files = {}
    try:
        h5_files['E'] = h5py.File(E_path, 'a')
        h5_files['N'] = h5py.File(N_path, 'a')
        h5_files['Z'] = h5py.File(Z_path, 'a')

        # Define dataset names
        dset_E_name = f'{dataset_name}_E_data'
        dset_N_name = f'{dataset_name}_N_data'
        dset_Z_name = f'{dataset_name}_Z_data'
        
        # Helper function to initialize resizable datasets
        def initialize_dset(h5_file, name, dtype):
            max_len = None # Use a large, but finite max size
            return h5_file.create_dataset(
                name, 
                (0,), 
                maxshape=(max_len,), 
                dtype=dtype, 
                chunks=True, 
                compression="gzip", 
                compression_opts=4
            )
        
        # --- Start Parallel Processing and Streaming ---
        with Pool(n_processes) as pool:
            results_iterator = pool.imap_unordered(worker_func, task_iterable)
            
            processed_count = 0
            # Wrap the iterator with tqdm to show streaming progress
            for idx, evi, E_data, N_data, Z_data in tqdm(
                results_iterator,
                total=len(ev_list),
                desc=f"Streaming {dataset_name} Events to HDF5 Files",
                mininterval=0.5
            ):
                if E_data is None:
                    continue # Skip event if worker failed
                
                # --- Stream I/O in Main Thread ---
                data_len = E_data.shape[0]
                data_dtype = E_data.dtype
                
                # Check for first run to initialize datasets
                if dset_E_name not in h5_files['E']:
                    dset_E = initialize_dset(h5_files['E'], dset_E_name, data_dtype)
                    dset_N = initialize_dset(h5_files['N'], dset_N_name, data_dtype)
                    dset_Z = initialize_dset(h5_files['Z'], dset_Z_name, data_dtype)
                else:
                    dset_E = h5_files['E'][dset_E_name]
                    dset_N = h5_files['N'][dset_N_name]
                    dset_Z = h5_files['Z'][dset_Z_name]

                # Get current size (all three datasets should have the same size)
                current_size = dset_E.shape[0]
                new_size = current_size + data_len

                # 1. Resize and Write E-data
                dset_E.resize(new_size, axis=0)
                dset_E[current_size:new_size] = E_data
                
                # 2. Resize and Write N-data
                dset_N.resize(new_size, axis=0)
                dset_N[current_size:new_size] = N_data
                
                # 3. Resize and Write Z-data
                dset_Z.resize(new_size, axis=0)
                dset_Z[current_size:new_size] = Z_data
                
                # --- Collect Metadata ---
                metadata_list.append((idx, evi, current_size))
                processed_count += 1
                
    finally:
        # Close all HDF5 files regardless of success or failure
        for f in h5_files.values():
            if f is not None:
                f.close()
                
    # --- Final I/O: Save Metadata CSV file ---
    print(f"Extraction complete. Saving {dataset_name} Metadata...")
    
    if processed_count == 0:
        print(f"No data was successfully processed for {dataset_name}.")
        return

    metadata_path = os.path.join(output_dir, f'{dataset_name}_metadata.csv')
    metadata_header = "original_index,event_id,array_start_index"
    
    with open(metadata_path, 'w') as f:
        f.write(metadata_header + "\n")
        for idx, evi, start_idx in metadata_list:
             f.write(f"{idx},{evi},{start_idx}\n")

    print(f"\nSuccessfully processed {processed_count} events.")
    print(f"Saved {dataset_name} Data (E, N, Z as separate HDF5 files) to:\n  E: {E_path}\n  N: {N_path}\n  Z: {Z_path}")
    print(f"Saved {dataset_name} Metadata to: {metadata_path}")
    print(f"{dataset_name} processing complete.")


if __name__ == '__main__':
    # You must replace HDF5_FILE_PATH and EV_LIST with your actual variables/paths.
    if HDF5_FILE_PATH == 'path/to/your/dtfl_file.h5':
         print("Warning: Please update HDF5_FILE_PATH and EV_LIST in the script before running.")
    
    EV_LIST = pd.read_csv("/Users/mabelwylie/Documents/EQ-AUDIO-DSL/raw_data/selected_event_list.csv")['trace_name'].tolist()  # Load your actual event IDs here
    # --- 1. Perform the 90/10 split on the event IDs ---
    train_ev_list, test_ev_list = train_test_split(
        EV_LIST, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

    print(f"Total events: {len(EV_LIST)}")
    print(f"Train events: {len(train_ev_list)}")
    print(f"Test events: {len(test_ev_list)}")
    print("-" * 30)

    # --- 2. Run the extraction pipeline for the training set ---
    run_extraction_pipeline(train_ev_list, HDF5_FILE_PATH, OUTPUT_DIR, N_PROCESSES, 'train')

    # --- 3. Run the extraction pipeline for the testing set ---
    run_extraction_pipeline(test_ev_list, HDF5_FILE_PATH, OUTPUT_DIR, N_PROCESSES, 'test')