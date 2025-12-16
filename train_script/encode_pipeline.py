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

import logging
from pyeclib.ec_iface import ECDriver
import argparse
import os
import psutil
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from mmap import mmap, ACCESS_READ
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np


# 1.Combine all data
# 2.Split the data into smaller block
parser = argparse.ArgumentParser(description='Encoder for PyECLib.')
parser.add_argument('-k', default=4, type=int, help='number of data elements')
parser.add_argument('-m', default=1, type=int, help='number of parity elements')
parser.add_argument('-ec_type', default="isa_l_rs_vand", help='EC algorithm used')
parser.add_argument('-file_dir', default="", help='directory with the files')
parser.add_argument('-fragment_dir', default="", help='directory to drop encoded fragments')
parser.add_argument('-output_path', default="", help='directory to acquire encoded fragments for decoding')

_ec_driver = None
args = parser.parse_args()
def setup_logger():
    logger = logging.getLogger("Checkpoint")
    if not logger.hasHandlers():  
        fh = logging.FileHandler('Checkpoint.log',encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s', '%H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    return logger

def init_worker(k, m, ec_type):
    global _ec_driver
    _ec_driver = ECDriver(k=k, m=m, ec_type=ec_type)

def encode_mmap_chunk(chunk_idx, file, chunk_size):
    with open(args.file_dir+file+'.pt', "rb") as f:
        mm = mmap(f.fileno(), 0, access=ACCESS_READ)
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(mm))
        # if start + chunk_size > len(mm) and len(mm) % 4 != 0:
        #     chunk_data = mm[start:end] + b'\0'*(start + chunk_size - len(mm))
        #     fragments = _ec_driver.encode(chunk_data)
        # else:
        fragments = _ec_driver.encode(mm[start:end])
        
        for i, fragment in enumerate(fragments):
            fragment_filename = f'{file}_chunk{chunk_idx}_fragment{i}.pt'
            with open(os.path.join(args.fragment_dir, fragment_filename), 'wb') as f:
                f.write(fragment)
        mm.close()

def main():
    print("k = %d, m = %d" % (args.k, args.m))
    print("ec_type = %s" % args.ec_type)
    if not os.path.exists(args.fragment_dir):
        os.makedirs(args.fragment_dir, exist_ok=True)

    ec_queue = multiprocessing.Queue()
    ec_process = multiprocessing.Process(target=asyn_ec, args=(ec_queue, ))
    ec_process.start()
    start_encode = time.time()
    for j in range(4):
        m_file = f"model_0_slice{j}"
        op_file = f"optimezer_0_slice{j}"
        ec_queue.put(m_file)
        ec_queue.put(op_file)

    ec_queue.put(None)
    ec_process.join()
    print("----------------clean!!!!-----------------")
    end = time.time()
    print(f"Total encode time: {end - start_encode:.2f} seconds")

def asyn_ec(ec_queue):
    while True:
        file = ec_queue.get()
        if file is None:
            break
        start_encode = time.time()
        with open(args.file_dir+file+'.pt', "rb") as f:
            mm = mmap(f.fileno(), 0, access=ACCESS_READ)
            chunk_size = 128 * 1024 * 1024  # 16MB
            total_chunks = (len(mm) + chunk_size -1) // chunk_size
            num_workers = min(int(os.cpu_count()/8), total_chunks)
            print(f"Chunk size: {chunk_size} ||| Total chunks: {total_chunks} ||| Using workers: {num_workers}")
            with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(args.k, args.m, args.ec_type)) as pool:
                tasks = [(i, file, chunk_size) for i in range(total_chunks)]
                for i in range(0, len(tasks), num_workers):
                    batch = tasks[i:i + num_workers]
                    pool.starmap_async(encode_mmap_chunk, batch)
                pool.close()
                pool.join()
            mm.close()
        end = time.time()
        print(f"{file} encode time: {end - start_encode:.2f} seconds")

if __name__ == "__main__":
    main()