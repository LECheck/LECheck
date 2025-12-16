# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit 6bd01c4
"""
import logging
from pyeclib.ec_iface import ECDriver
from mmap import mmap, ACCESS_READ
from collections import OrderedDict
import subprocess
import time
import torch
from .multi_tensor_apply import MultiTensorApply
import os
import fcntl
import json
import shutil
import datetime
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import FusedAdamBuilder

torch.multiprocessing.set_start_method('spawn', force=True) #
multi_tensor_applier = MultiTensorApply(2048 * 32)
loggername = "Checkpoint_1.log" 
_ec_driver = None

class FusedAdam(torch.optim.Optimizer): 
    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` may be used with or without Amp.  If you wish to use :class:`FusedAdam` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.  These additional arguments
        are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params,
                 module=None,
                 args=None,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 adam_w_mode=True,
                 weight_decay=0.,
                 amsgrad=False,
                 set_grad_none=True,
                 queue=None):

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        fused_adam_cuda = FusedAdamBuilder().load()
        # Skip buffer
        self._dummy_overflow_buf = get_accelerator().IntTensor([0])
        self.multi_tensor_adam = fused_adam_cuda.multi_tensor_adam

        # save process
        self.ckpt_run = False
        self.module = module
        self.logger = setlog()
        if args != None and args.ckpt_run == True:
            self.ckpt_run = args.ckpt_run
            self.ec_run = args.ec_run
            self.ckpt_freq = args.freq
            self.ckpt_home = args.cpu_home
            self.ckpt_steps = 0
            self.param_mappings = {}
            if getrank() == 0:
                path = os.path.join(self.ckpt_home, f"counts")
                with open(path, "w") as f:
                    fcntl.flock(f, fcntl.LOCK_EX) 
                    data = {}
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)
                self.logger.info(f"<Rank{getrank()}/{getworldsize()}> counts file init OK!! path is \"{path}\"")
            self._parameter_names = {v: k for k, v in sorted(list(self.module.named_parameters()))}
            self.checkpoint_lock = torch.multiprocessing.Lock()
            self.save_queue = torch.multiprocessing.Queue()
            self.ec_queue = torch.multiprocessing.Queue()
            self.save_process = torch.multiprocessing.Process(target=asyn_save, args=(self.ec_queue, self.save_queue, self.module, len(self.param_groups), self.ckpt_home, self.checkpoint_lock))
            self.save_process.start()
            if self.ec_run:
                self.ec_param = (args.k, args.m, args.ec_type)
                self.ec_fragment_dir = args.fragment_dir
                self.ec_process = torch.multiprocessing.Process(target=asyn_ec, args=(self.ec_queue, self.ec_param, self.ckpt_home, self.ec_fragment_dir, len(self.param_groups)))
                self.ec_process.start()

    def zero_grad(self):
        
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.'
            )
        loss = None

        if closure is not None:
            loss = closure()

        index = 0
        start_index = 0
        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' not in group:
                group['step'] = 0

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                if p.grad is None or torch.all(p.grad == 0):
                    self.logger.info("torch.all(p.grad == 0) torch.all(p.grad == 0) torch.all(p.grad == 0) torch.all(p.grad == 0)!!!")
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # DeepSpeed ZeRO 3 processes each subgroup a time, so we need to keep tracking step count for each tensor separately.
                    # While this is not an issue for ZeRO 1 & 2, since they apply a single optimization step to the whole param group at the same time.
                    # In order to keep backward compatibility for the existing checkpoints, we use group['state'] to initialize state['step'] if it exists.
                    state['step'] = group.get('step', 0)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                    v_bf.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16, bf16 and fp32.')

            if self.ckpt_run:
                if self.ckpt_steps == 1:
                    ss2 = time.time()
                    self.checkpoint_lock.acquire()
                    self.checkpoint_lock.release()
                    ss1 = time.time()
                    self.logger.info("<Rank{}/{}> <iteration{}> ckpt blocking cost ——————> {}".format(getrank(), getworldsize(), state['step'], ss1-ss2))

            if len(g_16) > 0:
                state['step'] += 1
                batch_size = len(g_16) // 4
                for i in range(4):
                    start = i * batch_size
                    end = (i + 1) * batch_size if i < 3 else len(g_16)
                    g_batch = g_16[start:end]
                    p_batch = p_16[start:end]
                    m_batch = m_16[start:end]
                    v_batch = v_16[start:end]
                    multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_batch, p_batch, m_batch, v_batch],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

            if len(g_bf) > 0:
                state['step'] += 1
                batch_size = len(g_bf) // 4
                for i in range(4):
                    start = i * batch_size
                    end = (i + 1) * batch_size if i < 3 else len(g_bf)
                    g_batch = g_bf[start:end]
                    p_batch = p_bf[start:end]
                    m_batch = m_bf[start:end]
                    v_batch = v_bf[start:end]
                    multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_batch, p_batch, m_batch, v_batch],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

            if len(g_32) > 0:
                state['step'] += 1
                batch_size = len(g_32) // 4
                for i in range(4):
                    start = i * batch_size
                    end = (i + 1) * batch_size if i < 3 else len(g_32)
                    g_batch = g_32[start:end]
                    p_batch = p_32[start:end]
                    m_batch = m_32[start:end]
                    v_batch = v_32[start:end]
                    multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_batch, p_batch, m_batch, v_batch],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])
                    # checkpointing parameter/state
                    if self.ckpt_run and getrank() == i and self.ckpt_freq == self.ckpt_steps:
                        self.logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{state['step']-1}> ckpt in memory is beginning!")
                        # model.state_dict()
                        mod = OrderedDict({self._parameter_names[p] : p for p in group['params'][start:end]})
                        self.save_queue.put((mod, index, True, True, state['step']-1))

                        # optimizer.state_dict()
                        ops =  {}
                        def pack_group(gr):
                            nonlocal start_index
                            packed = {k: v for k, v in gr.items() if k != 'params'}
                            self.param_mappings.update({id(p): i for i, p in enumerate(gr['params'], start_index)
                                                if id(p) not in self.param_mappings})
                            packed['params'] = [self.param_mappings[id(p)] for p in gr['params']]
                            start_index += len(packed['params'])
                            return packed
                        if getrank() == 1:
                            opsg =  {}
                            opsg['param_groups'] = pack_group(group)
                            self.save_queue.put((opsg, index, False, False, state['step']-1))
                        else:
                            pack_group(group)
                        ops['state'] = {self.param_mappings[id(p)] : self.state[p] for p in group['params'][start:end]}
                        self.save_queue.put((ops, index, False, True, state['step']-1))
            index = index + 1

        # Calculate frequency
        if self.ckpt_run:
            if self.ckpt_freq == self.ckpt_steps: self.ckpt_steps = 1
            else: self.ckpt_steps += 1
    
        return loss
    
    def op_exit(self):
        """Clean up"""
        if self.ckpt_run:
            self.save_queue.put((None, None, None, None, None))
            self.save_process.join()
            if self.ec_run:
                self.ec_queue.put((None, None, None))
                self.ec_process.join()
        self.logger.info(f"<Rank{getrank()}/{getworldsize()}> newopt clean!!!!")
    
def asyn_save(ec_queue, save_queue, model, num_param_group, ckpt_home, checkpoint_lock):
    model_num = 0
    opt_num = 0
    logger = setlog()
    while True:
        # start_save = time.time()
        (s, index, flag, isstate, iteration) = save_queue.get()
        if s is None:
            break
        path = os.path.join(ckpt_home, f"iteration{iteration}/")
        if flag:
            if model_num == 0:
                checkpoint_lock.acquire()
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                # checkpointing buffer
                if getrank() == 0:
                    memo = set()
                    model_buffer = OrderedDict()
                    for module_prefix, module in model.named_modules():
                        members = module._buffers.items()
                        for k, v in members:
                            if v is None or v in memo or k in module._non_persistent_buffers_set:
                                continue
                            memo.add(v)
                            n = module_prefix + ('.' if module_prefix else '') + k
                            model_buffer[n] = v
                    torch.save(model_buffer, f"{path}model_buffer.pt")
                    # logger.info(f"Saved {ckpt_home}model_buffer.pt at {datetime.datetime.now()}")
            model_num += 1
            torch.save(s, f"{path}model_{index}_slice{getrank()}.pt")
            ec_queue.put((f"model_{index}_slice{getrank()}", True, iteration))
            # logger.info(f"Saved {ckpt_home}model_{index}_slice{getrank()}.pt at {datetime.datetime.now()}")
        else:
            if not isstate:
                torch.save(s, f"{path}optimezer_{index}_param_groups.pt")
            else:
                opt_num += 1
                torch.save(s, f"{path}optimezer_{index}_slice{getrank()}.pt")
                ec_queue.put((f"optimezer_{index}_slice{getrank()}", False, iteration))
                # logger.info(f"Saved {ckpt_home}optimezer_{index}_slice{getrank()}.pt at {datetime.datetime.now()}")
        if model_num == num_param_group and opt_num == num_param_group:
            model_num = 0
            opt_num = 0
            checkpoint_lock.release()
            logger.info("<Rank{}/{}> <iteration{}> ckpt finished!! in memory cost ——————> {}".format(getrank(), getworldsize(), iteration, cpu_memory_report(ckpt_home)))

def init_worker(k, m, ec_type):
    global _ec_driver
    _ec_driver = ECDriver(k=k, m=m, ec_type=ec_type)

def asyn_ec(ec_queue, ec_param, ckpt_home, ec_home, num_param_group):
    if not os.path.exists(ec_home):
        os.makedirs(ec_home, exist_ok=True)
    model_num = 0
    opt_num = 0
    lastiter = 0
    logger = setlog()
    (k, m, ec_type) = ec_param
    init_worker(k, m, ec_type)
    while True:
        (file, flag, iteration) = ec_queue.get()
        if file is None:
            break
        start_encode = time.time()
        pathec = os.path.join(ec_home, f"iteration{iteration}/")
        if model_num == 0:
            time.sleep(0.6)
            if not os.path.exists(pathec):
                os.makedirs(pathec, exist_ok=True)
        if flag:    model_num += 1
        else:   opt_num += 1
        pathckpt = os.path.join(ckpt_home, f"iteration{iteration}/{file}.pt")
        with open(pathckpt , "rb") as f:
            mm = mmap(f.fileno(), 0, access=ACCESS_READ)
            chunk_size = 1024 * 1024 * 1024  # 1G
            total_chunks = (len(mm) + chunk_size -1) // chunk_size
            # num_workers = min(int(os.cpu_count()/48), total_chunks)
            # logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> Chunk size: {chunk_size} ||| Total chunks: {total_chunks} ||| Using workers: {num_workers}")
            mm.close()
        # logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> {total_chunks}")
        for i in range(total_chunks):
            encode_mmap_chunk(i, file, chunk_size, pathckpt, pathec)
            
        end = time.time()
        logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> {file} Encode Mission OK!  ——————> time: {end - start_encode:.2f} seconds")
        if model_num == num_param_group and opt_num == num_param_group:
            model_num = 0
            opt_num = 0
            fflag = False
            with open(os.path.join(ckpt_home, f"counts"), "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX) 
                data = json.load(f)
                key = str(iteration)
                if key in data:
                    data[key] += 1
                else:
                    data[key] = 1
                if data[key] == 4 and lastiter != 0:
                    fflag = True
                    del data[key]
                f.seek(0)
                json.dump(data, f)
                f.truncate()
                fcntl.flock(f, fcntl.LOCK_UN) 
            if fflag:
                shutil.rmtree(os.path.join(ckpt_home, f"iteration{lastiter}"))
                shutil.rmtree(os.path.join(ec_home, f"iteration{lastiter}"))
                logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> Have cleared old ckpt and ec_parity!!")
            lastiter = iteration
            logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> All Encode Mission OK!!")

def encode_mmap_chunk(chunk_idx, file, chunk_size, ckpt_home, ec_home):
    with open(ckpt_home, "rb") as f:
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
            with open(os.path.join(ec_home, fragment_filename), 'wb') as f:
                f.write(fragment)
        mm.close()

def cpu_memory_report(ckpt_home):
    result = subprocess.run([f"du -sh {ckpt_home}/"], capture_output=True, text=True, shell=True)
    return result.stdout.strip()

def getrank():
    return int(os.environ.get("RANK", "0"))

def getworldsize():
    return int(os.environ.get("WORLD_SIZE", "1"))

def has_d_state_processes():
    result = subprocess.run(
        ["ps", "-A", "-ostat,pid,cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    d_processes = subprocess.run(
        ["grep", "^D"],
        input=result.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    return bool(d_processes.stdout.strip())

def setlog():
    logger = logging.getLogger("Checkpoint")
    if not logger.hasHandlers():  
        fh = logging.FileHandler(loggername, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s', '%H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger