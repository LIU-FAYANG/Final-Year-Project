import collections
import inspect
import math
import sys
import copy
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import io
import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy
# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)

from datasets import load_dataset, Dataset
from scipy.stats import spearmanr

import re

class CLTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
        stsb_score_range: Optional[List[float]]=None,
        dev_file:Optional[str]=None,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        # tasks = ["SICKRelatedness"]
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        metrics = {"eval_stsb_spearman": stsb_spearman,
                   "eval_sickr_spearman": sickr_spearman,
                   "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}


        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer

        if self.args.dev_file is not None:
            if not hasattr(self, "dev_dataset"):
                self.dev_dataset = load_dataset("csv", data_files=self.args.dev_file)["train"]

            def encode(sentences):
                batch = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                )
                for k in batch:
                    batch[k] = batch[k].to(self.args.device)
                with torch.no_grad():
                    outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                    pooler_output = outputs.pooler_output
                return pooler_output.cpu()

            bs = 128
            n_steps = len(self.dev_dataset)//bs
            if len(self.dev_dataset)%bs!=0:
                n_steps+=1
            column_names = self.dev_dataset.column_names
            sims = []
            for step in range(n_steps):
                e1 = encode(self.dev_dataset[column_names[0]][step*bs:(step+1)*bs])
                e2 = encode(self.dev_dataset[column_names[1]][step*bs:(step+1)*bs])
                sims+=torch.cosine_similarity(e1, e2, dim=-1).numpy().tolist()
            metrics["eval_dev_spearman"] = spearmanr(self.dev_dataset[column_names[-1]], sims)[0]

        metrics["d_loss"] = self.d_loss
        self.log(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save.
        assert _model_unwrap(model) is self.model, "internal model should be a reference to self.model"
        '''
        print("load previous checkpoint as distillation teacher!!!!!!")
        best_model = copy.copy(self.model)
        self.teachers = best_model
        '''

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                '''
                previous = None 
                teacher_folder = self.model_args.distill_teacher
                for f in teacher_folder:
                    source_path = f
                    destination_path = previous 
                    if destination_path:
                        shutil.rmtree(destination_path)

                    # Overwrite files in distill_teacher2 with files from distill_teacher1
                    if previous:
                        shutil.copytree(source_path, destination_path)
                        print(source_path + " copy to " + destination_path)
                    previous = f
                    '''

                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                print("saving best ckpt to OUTPUT DIR!!!!!!!!")
                # Only save model when it is the best one
                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                # Save optimizer and scheduler
                if self.sharded_dpp:
                    self.optimizer.consolidate_state_dict()

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        reissue_pt_warnings(caught_warnings)
                elif self.is_world_process_zero() and not self.deepspeed:
                    # deepspeed.save_checkpoint above saves model/optim/sched
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)

                # Save the Trainer state
                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
                '''
                #print(self.state.global_step)
                if self.state.global_step >= self.model_args.CL_steps and self.model_args.distill_teacher is not None:
                    #packing distillation teachers 
                    #when the best checkpoint is saved, shallow copy it and load it as the teache model.
                    print("deepcopy best model and load best model as distillation teacher!!!!!!")
                    
                    #best_model = copy.deepcopy(self.model)
                    #self.teachers = nn.ModuleList([best_model])
                    #for p in self.model_args.distill_teacher:
                    #    print(p)
                    
                    self.teachers = nn.ModuleList([AutoModel.from_pretrained(p).eval() for p in self.model_args.distill_teacher])
                    if self.args.n_gpu==1:

                        #self.teachers = self.teachers.to("cuda")
                        for i in range(len(self.teachers)):
                            self.teachers[i] = self.teachers[i].to("cuda")
                        
                    else:
                        for i in range(len(self.teachers)):
                            device = torch.device(f"cuda:{i%(self.args.n_gpu -self.args.n_gpu_for_training)+self.args.n_gpu_for_training}")
                            self.teachers[i] = self.teachers[i].to(device)
                            '''
                    
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune

                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                self.store_flos()

            self.save_model(output_dir)
            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)

            # Save optimizer and scheduler
            if self.sharded_dpp:
                self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                # deepspeed.save_checkpoint above saves model/optim/sched
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)


            # Save the Trainer state
            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # Maybe delete some older checkpoints.
            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.

        The main difference between ours and Huggingface's original implementation is that we
        also load model_args when reloading best checkpoints for evaluation.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)

            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)

            self.model = model
            self.model_wrapped = model

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        model = self.model_wrapped

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(self.args.n_gpu_for_training)))


        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_dpp:
            model = ShardedDDP(model, self.optimizer)
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (self.args.n_gpu_for_training if self.args.local_rank != -1 else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            inputs = None
            last_inputs = None
            device0 = torch.device("cuda:0")
            #teacher_hidden_size = self.teachers.config.hidden_size
            teacher_hidden_size = [t.config.hidden_size for t in self.teachers]
            if len(set(teacher_hidden_size))==1:
                homo=True
            else:
                homo=False
                print("using mixed model for distillation")
                n_base, n_large = (torch.tensor(teacher_hidden_size)==768).sum().to(int), (torch.tensor(teacher_hidden_size)==1024).sum().to(int)
            odd = False
            if len(teacher_hidden_size) % 2 != 0:
                odd = True

            # check if all the teachers are homogeneous
            for step, inputs in enumerate(epoch_iterator):

                #save ckpt to distill_teacher folder, every 125 steps ot 500 steps
                if step >= (self.model_args.CL_steps - 500):
                    if step % 125 == 0: 
                        #update distill teacher folders.... 
                        previous = None 
                        teacher_folder = self.model_args.distill_teacher
                        print("Organizing Disitill Teachers.....")
                        for f in teacher_folder:
                            source_path = f
                            destination_path = previous 
                            if destination_path:
                                shutil.rmtree(destination_path)

                            # Overwrite files in distill_teacher2 with files from distill_teacher1
                            if previous:
                                shutil.copytree(source_path, destination_path)
                                print(source_path + " copy to " + destination_path)
                            previous = f
                        
                        output_dir = self.model_args.distill_teacher[-1]
                        print("saving to distill teacher: ")
                        print(output_dir)
                        self.save_model(output_dir)
                        if self.deepspeed:
                            self.deepspeed.save_checkpoint(output_dir)

                        # Save optimizer and scheduler
                        if self.sharded_dpp:
                            self.optimizer.consolidate_state_dict()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            with warnings.catch_warnings(record=True) as caught_warnings:
                                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                reissue_pt_warnings(caught_warnings)
                        elif self.is_world_process_zero() and not self.deepspeed:
                            # deepspeed.save_checkpoint above saves model/optim/sched
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            with warnings.catch_warnings(record=True) as caught_warnings:
                                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            reissue_pt_warnings(caught_warnings)

                        # Save the Trainer state
                        if self.is_world_process_zero():
                            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


                        if self.state.global_step >= self.model_args.CL_steps and self.model_args.distill_teacher is not None:
                            #packing distillation teachers 
                            #when the best checkpoint is saved, shallow copy it and load it as the teache model.
                            print("Load best model from DISTILL TEACHER DIR as distillation teacher!!!!!!")
                            
                            #best_model = copy.deepcopy(self.model)
                            #self.teachers = nn.ModuleList([best_model])
                            #for p in self.model_args.distill_teacher:
                            #    print(p)
                            
                            self.teachers = nn.ModuleList([AutoModel.from_pretrained(p).eval() for p in self.model_args.distill_teacher])
                            if self.args.n_gpu==1:

                                #self.teachers = self.teachers.to("cuda")
                                for i in range(len(self.teachers)):
                                    self.teachers[i] = self.teachers[i].to("cuda")
                                
                            else:
                                for i in range(len(self.teachers)):
                                    device = torch.device(f"cuda:{i%(self.args.n_gpu -self.args.n_gpu_for_training)+self.args.n_gpu_for_training}")
                                    self.teachers[i] = self.teachers[i].to(device)
                
                #copy warm-up finished model(pure CL) to another folder for further evaluation 
                if step == self.model_args.CL_steps:
                    source_dir = self.args.output_dir
                    destination_dir = self.args.output_dir + "/warmup_ckpt"

                    os.makedirs(destination_dir, exist_ok=True)
                    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
                
                #distillation starts, load firsts teacher model 
                if step == self.model_args.CL_steps and self.model_args.distill_teacher is not None:
                    #packing distillation teachers 
                    #when the best checkpoint is saved, shallow copy it and load it as the teache model.
                    print("Distillation starts, load previous best checkpoint as distillation teacher!!!!!!")
                    
                    self.teachers = nn.ModuleList([AutoModel.from_pretrained(p).eval() for p in self.model_args.distill_teacher])
                    if self.args.n_gpu==1:

                        #self.teachers = self.teachers.to("cuda")
                        for i in range(len(self.teachers)):
                            self.teachers[i] = self.teachers[i].to("cuda")
                        
                    else:
                        for i in range(len(self.teachers)):
                            device = torch.device(f"cuda:{i%(self.args.n_gpu -self.args.n_gpu_for_training)+self.args.n_gpu_for_training}")
                            self.teachers[i] = self.teachers[i].to(device)

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                ### embedding the teacher logits
                sim_t = []
                teacher_hidden_size = []
                sim_t = torch.zeros([self.args.per_device_train_batch_size, self.args.per_device_train_batch_size], device=device0)
                with torch.no_grad():
                    for i, teacher in enumerate(self.teachers):
                        teacher = teacher.eval()
                        outputs_t = teacher(inputs["input_ids"][:, 0].to(teacher.device),
                                            inputs["attention_mask"][:, 0].to(teacher.device),
                                            inputs["token_type_ids"][:, 0].to(teacher.device) if "token_type_ids" in inputs else None)
                        e = outputs_t.last_hidden_state[:, 0]
                        sim_tt = torch.cosine_similarity(e.unsqueeze(0), e.unsqueeze(1), dim=-1).to(device0)
                        if homo:
                            sim_t += sim_tt / len(self.teachers)   # (b,b,2) or (b,b)
                        else:
                            if teacher.config.hidden_size == 768: #base model
                                sim_t += self.model_args.distill_alpha *  sim_tt/ n_base
                            else: #large model
                                sim_t += (1-self.model_args.distill_alpha) * sim_tt/ n_large
                    
                    if odd and i==(len(self.teachers)-1):
                        sim_t = (1-self.args.tt)*sim_t+self.args.tt*sim_tt
                        
                inputs["sim_t"] = sim_t.cpu()

                if self.args.stopping_steps is not None and step>self.args.stopping_steps:
                    break


                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        self.temp_loss = self.training_step(model, inputs)
                        tr_loss += self.temp_loss
                else:
                    self.temp_loss = self.training_step(model, inputs)
                    tr_loss += self.temp_loss

                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()

                    model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break


            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step
                           , self._total_loss_scalar / self.state.global_step, metrics)
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size*self.args.n_gpu_for_training, #use trainer.args.n_gpu_for_training gpus for training
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        self.d_loss = outputs["d_loss"].item()
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            return self.label_smoother(outputs, inputs["labels"])
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            return outputs["loss"] if isinstance(outputs, dict) else outputs[0]



class AutoDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)

    def __getitem__(self, item):
        return self.dataset[item % self.length]

    def __len__(self):
        return 1000* self.length




