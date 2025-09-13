# Copyright (c) 2013, Kevin Greenan (kmgreen2@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.  THIS SOFTWARE IS
# PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
# NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pyeclib.ec_iface import ECDriver
import argparse
import os
import psutil
import time
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)
from functools import partial
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from mmap import mmap, ACCESS_READ
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np

# 1. Combine all data
# 2. Split the data into smaller blocks
parser = argparse.ArgumentParser(description='Encoder for PyECLib.')
parser.add_argument('-k', default=4, type=int, help='number of data elements')
parser.add_argument('-m', default=1, type=int, help='number of parity elements')
parser.add_argument('-ec_type', default="isa_l_rs_vand", help='EC algorithm used')
parser.add_argument('-file_dir', default="/data/lff/eccheck/", help='directory with the files')
parser.add_argument('-filenames', nargs='+', help='files to decode')
parser.add_argument('-filenumber', default=4, type=int, help='number of files to decode')
parser.add_argument('-fragment_dir', default="", help='directory to drop encoded fragments')
parser.add_argument('-output_path', default="", help='directory to acquire encoded fragments for decoding')
parser.add_argument('-output', default="")

_ec_driver = None
args = parser.parse_args()

def init_worker(k, m, ec_type):
    """Initialize the EC driver in worker processes"""
    global _ec_driver
    _ec_driver = ECDriver(k=k, m=m, ec_type=ec_type)

def decode():
    """Decode data fragments using parallel processing"""
    # Collect fragment information
    chunk_frags = defaultdict(list)
    for fname in os.listdir(args.fragment_dir):
        if not fname.endswith('.pt'):
            continue
        parts = fname.split('_')
        if len(parts) != 4:
            continue
        chunk_id = int(parts[1])
        frag_id = int(parts[3].split('.')[0])
        chunk_frags[chunk_id].append( (frag_id, os.path.join(args.fragment_dir, fname)) )

    # Prepare valid chunk list
    valid_chunks = [cid for cid, frags in chunk_frags.items() if len(frags)>=args.k]

    # Multi-process decoding (initialize driver with initializer)
    with multiprocessing.Pool(
        initializer=init_worker,
        initargs=(args.k, args.m, args.ec_type)
    ) as pool:
        results = pool.map(decode_chunk_wrapper, [
            (cid, [f[1] for f in chunk_frags[cid][:args.k]])  # Only pass chunk ID and path list
            for cid in valid_chunks
        ])

    # Reassemble data
    with open(args.output, 'wb') as fout:
        for chunk_id, data in sorted(zip(valid_chunks, results), key=lambda x: x[0]):
            if data is None:
                raise ValueError(f"Chunk {chunk_id} decode failed")
            fout.write(data)

def decode_chunk_wrapper(args):
    """Wrapper function to unpack arguments (since pool.map doesn't support multiple arguments)"""
    return decode_chunk(*args)

def decode_chunk(chunk_id, frag_paths):
    """Actual decoding function (using process-local ECDriver)"""
    try:
        # Use global variable _local_ec_driver (initialized in child process)
        global _local_ec_driver
        
        # Asynchronously read fragments
        frag_data = {}
        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(read_fragment, path): idx
                for idx, path in enumerate(frag_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                frag_data[idx] = future.result()

        # Decode
        return _local_ec_driver.decode(
            [frag_data[i] for i in range(len(frag_data))],
            list(frag_data.keys())
        )
    except Exception as e:
        print(f"Chunk {chunk_id} error: {str(e)}")
        return None

def decode_parallel():
    """Decode data fragments using parallel processing with ProcessPoolExecutor"""
    # Initialize global driver
    start = time.time()
    ec_driver = ECDriver(k=args.k, m=args.m, ec_type=args.ec_type)

    # Discover available fragments (optimized version)
    chunk_map = defaultdict(dict)
    frag_files = [f for f in os.listdir(args.fragment_dir) if f.endswith('.pt')]
    
    # Use numpy to accelerate analysis
    pattern = np.core.defchararray.split(frag_files, '_')
    valid_mask = np.array([len(p) == 4 for p in pattern])
    valid_files = np.array(frag_files)[valid_mask]
    
    chunks = np.core.defchararray.replace(
        np.array([p[1] for p in pattern if len(p)==4]), 'chunk', '').astype(int)
    frag_ids = np.core.defchararray.replace(
        np.array([p[3].split('.')[0] for p in pattern if len(p)==4]), 'fragment', '').astype(int)
    
    for i in range(len(valid_files)):
        chunk_map[chunks[i]][frag_ids[i]] = os.path.join(args.fragment_dir, valid_files[i])

    # Generate decoding tasks
    decode_tasks = []
    for chunk_id, frags in chunk_map.items():
        if len(frags) >= args.k:
            decode_tasks.append((chunk_id, sorted(frags.items()[:args.k])))  # Take first k valid fragments
        else:
            print(f"Chunk {chunk_id} has only {len(frags)} fragments")

    start_decode = time.time()
    # Multi-process decoding
    with ProcessPoolExecutor(max_workers=os.cpu_count()*2) as executor:
        future_map = {}
        for chunk_id, frags in decode_tasks:
            future = executor.submit(
                decode_chunk_worker,
                chunk_id,
                [f[1] for f in frags],  # Fragment path list
                args.k,
                args.ec_type
            )
            future_map[future] = chunk_id

        # Collect results
        decoded_chunks = {}
        for future in as_completed(future_map):
            chunk_id = future_map[future]
            try:
                data = future.result()
                decoded_chunks[chunk_id] = data
            except Exception as e:
                print(f"Chunk {chunk_id} decode failed: {str(e)}")

    # Reassemble in order
    with open(args.output, 'wb') as f_out:
        for chunk_id in sorted(decoded_chunks.keys()):
            f_out.write(decoded_chunks[chunk_id])
    print(f"Total encode time: {end - start_decode:.2f} seconds")
    print(f"Total time: {end - start:.2f} seconds")

def decode_chunk_worker(chunk_id, frag_paths, k, ec_type):
    """Worker function for decoding chunks (runs independently per CPU core)"""
    # Initialize within process (avoid shared objects)
    ec_driver = ECDriver(k=k, m=len(frag_paths)-k, ec_type=ec_type)
    
    # Asynchronously read fragments in parallel
    frag_data = []
    with ThreadPoolExecutor(max_workers=4) as io_executor:
        read_futures = []
        for path in frag_paths:
            future = io_executor.submit(read_fragment, path)
            read_futures.append(future)
        
        for future in as_completed(read_futures):
            frag_data.append(future.result())

    # Decoding computation
    valid_frags = {i: frag_data[i] for i in range(len(frag_data))}
    return ec_driver.decode(list(valid_frags.values()), list(valid_frags.keys()))

def read_fragment(path):
    """Use memory mapping to accelerate fragment reading"""
    with open(path, 'rb') as f:
        mm = mmap(f.fileno(), 0, access=ACCESS_READ)
        data = bytes(mm)  # Copy to process memory
        mm.close()
    return data

def serial_decode():
    """Decode data fragments serially (without parallel processing)"""
    ec_driver = ECDriver(k=args.k, m=args.m, ec_type=args.ec_type)

    fragment_list = []
    chunk_map = defaultdict(dict)
    print(f"Scanning fragments in {args.fragment_dir}...")
    for filename in os.listdir(args.fragment_dir):
        if not filename.endswith('.pt'):
            continue
        
        # Parse filename format: prefix_chunkX_fragmentY.pt
        parts = filename.split('_')
        if len(parts) != 4:
            print(f"Ignoring malformed file: {filename}")
            continue
        
        try:
            chunk_id = int(parts[1].replace("chunk", ""))
            frag_id = int(parts[3].replace("fragment", "").split('.')[0])
            full_path = os.path.join(args.fragment_dir, filename)
            chunk_map[chunk_id][frag_id] = full_path
        except ValueError:
            print(f"Failed to parse chunk/fragment ID from {filename}")
            continue

    # Step 2: Validate fragment integrity
    valid_chunks = []
    for chunk_id, frags in chunk_map.items():
        if len(frags) < args.k:
            print(f"Chunk {chunk_id} has only {len(frags)} fragments (need at least {args.k})")
        else:
            valid_chunks.append(chunk_id)
    
    if not valid_chunks:
        raise ValueError("No valid chunks available for reconstruction")

    print(f"Found {len(valid_chunks)} valid chunks with sufficient fragments")
    restored_data = bytearray()
    for chunk_id in sorted(valid_chunks):
        frag_paths = []
        frag_ids = sorted(chunk_map[chunk_id].keys())[:args.k]  # Take first k available fragments
        
        for fid in frag_ids:
            frag_paths.append(chunk_map[chunk_id][fid])
        
        # Read fragment data
        frag_data = []
        for path in frag_paths:
            with open(path, 'rb') as f:
                frag_data.append(f.read())
        
        # Perform decoding
        try:
            decoded_data = ec_driver.decode(frag_data)
            restored_data.extend(decoded_data)
            print(f"Chunk {chunk_id} decoded successfully ({len(decoded_data)} bytes)")
        except Exception as e:
            print(f"Failed to decode chunk {chunk_id}: {str(e)}")
            raise

    # Step 5: Write final file
    with open(args.output_path, 'wb') as f:
        f.write(restored_data)
    
    print(f"Reconstruction completed. Saved to {args.output_path}")

if __name__ == "__main__":
    start = time.time()
    serial_decode()
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")