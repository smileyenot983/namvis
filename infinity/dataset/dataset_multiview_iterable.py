import glob
import os
import pickle
import random
import re
import time
from functools import partial
from os import path as osp
from typing import List, Tuple, Union
import json
import itertools
import concurrent.futures
from multiprocessing import cpu_count

import tqdm
import numpy as np
import torch
import pandas as pd
from PIL import Image as PImage
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as tdist
import torchvision

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, get_h_div_w_template2indices, h_div_w_templates
from infinity.utils.large_file_util import get_part_jsonls, split_large_txt_files

from pathlib import Path


import h5py 
def center_crop_to_tensor_pm1(pil_image, mid_reso: int, final_reso: int):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    Then to_tensor and normalize to [-1, 1]
    """
    while min(*pil_image.size) >= 2 * mid_reso:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PImage.BOX
        )
    
    if mid_reso == final_reso == pil_image.size[0] == pil_image.size[1]:
        im = to_tensor(pil_image)
    else:
        # resize the shorter edge to mid_reso
        scale = mid_reso / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PImage.LANCZOS
        )
        
        # crop the center out
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - final_reso) // 2
        crop_x = (arr.shape[1] - final_reso) // 2
        # return PImage.fromarray(arr[crop_y: crop_y + final_reso, crop_x: crop_x + final_reso])
        im = to_tensor(arr[crop_y: crop_y + final_reso, crop_x: crop_x + final_reso])
    
    return im.add(im).add_(-1)

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1)

def transform_wintr(pil_img, intr, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        scale = tgt_w / width
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        scale = tgt_h / height
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)



    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2

    K = torch.as_tensor(intr, dtype=torch.float32).clone()

    scale_x = resized_width / width
    scale_y = resized_height / height

    K[0, :] *= scale_x
    K[1, :] *= scale_y
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y
    K[0, :] /= tgt_w
    K[1, :] /= tgt_h
    K[2, :] = torch.tensor([0.0, 0.0, 1.0])

    new_intr = K

    # apply same rescale + crop to intrinsics
    # new_intr = torch.eye(3)
    # new_intr[0,0] = scale*intr[0,0] # / tgt_w
    # new_intr[1,1] = scale*intr[1,1] # / tgt_h

    #TODO:  
    # new_intr[0,2] = (scale*intr[0,2] - crop_x) # / tgt_w
    # new_intr[1,2] = (scale*intr[1,2] - crop_y) # / tgt_h

    # new_intr = torch.tensor([[1.0, 0.0, 0.5],
    #                          [0.0, 1.0, 0.5],
    #                          [0.0, 0.0, 1.0]])

    # print(f"scale: {scale}")
    # print(f"intr: {intr}")
    # print(f"new_intr: {new_intr}")

    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1), new_intr

def process_short_text(short_text):
    if '--' in short_text:
        processed_text = short_text.split('--')[0]
        if processed_text:
            short_text = processed_text
    return short_text


class MultiviewTestDataset(Dataset):
    def __init__(
        self, 
        meta_folder: str, 
        max_caption_len=512, 
        short_prob=0.2, 
        load_vae_instead_of_image=False,
        buffersize: int = 10000,
        seed: int = 0, 
        pn: str = '',
        online_t5: bool = True,
        batch_size: int = 2,
        num_replicas: int = 1, # 1,
        rank: int = 0, # 0
        dataloader_workers: int = 2,
        dynamic_resolution_across_gpus: bool = True,
        enable_dynamic_length_prompt: bool = True,
        N_views_src=0,
        N_views_tgt=0,
        identity_test=False,
        hdf5_path=None,
        **kwargs,
    ):

        self.meta_folder = meta_folder
        self.pn = pn
        self.online_t5 = online_t5
        self.buffer_size = buffersize
        self.num_replicas = num_replicas
        self.rank = rank
        self.worker_id = 0
        self.global_worker_id = 0
        self.dataloader_workers = max(1, dataloader_workers)
        self.max_caption_len = max_caption_len
        self.short_prob = short_prob
        self.load_vae_instead_of_image = load_vae_instead_of_image # set to false
        self.dynamic_resolution_across_gpus = dynamic_resolution_across_gpus
        self.enable_dynamic_length_prompt = enable_dynamic_length_prompt
        self.batch_size = batch_size
        # print(f'self.dynamic_resolution_across_gpus: {self.dynamic_resolution_across_gpus}')
        # print(f'self.enable_dynamic_length_prompt: {self.enable_dynamic_length_prompt}')
        # print(f'self.buffer_size: {self.buffer_size}')
        self.shuffle = True
        self.global_workers = self.num_replicas * self.dataloader_workers
        # self.h_div_w_template2generator, self.samples_div_gpus_workers_batchsize_2batches, total_samples = self.set_h_div_w_template2generator()
        # self.split_meta_files()
        self.seed = seed
        self.epoch_worker_generator = None
        self.epoch_global_worker_generator = None
        # self.set_epoch(0)
        # print(f'num_replicas: {num_replicas}, rank: {rank}, dataloader_workers: {dataloader_workers}, seed:{seed}, samples_div_gpus_workers_batchsize_2batches: {self.samples_div_gpus_workers_batchsize_2batches}')

        self.identity_test = identity_test
        self.hdf5_path = hdf5_path

        self.jsonl_path = glob.glob(osp.join(self.meta_folder, '*.jsonl'))[0]
        
        # print(f"self.jsonl_path: {self.jsonl_path}")
        self.test_paths, self.test_indices_src, self.test_indices_tgt = self.load_eval_config(self.jsonl_path)
        
        if N_views_src !=0 and N_views_tgt !=0:
            self.N_views_src = N_views_src
            self.N_views_tgt = N_views_tgt
            self.test_indices_src = [test_indices_i[:self.N_views_src] for test_indices_i in self.test_indices_src]
            self.test_indices_tgt = [test_indices_i[:self.N_views_tgt] for test_indices_i in self.test_indices_tgt]

        else:

            self.N_views_src = N_views_src
            self.N_views_tgt = N_views_tgt

        print(f"[MultiviewTestDataset]self.test_paths: {self.test_paths}")
        print(f"[MultiviewTestDataset]self.test_indices_src: {self.test_indices_src}")
        print(f"[MultiviewTestDataset]self.test_indices_tgt: {self.test_indices_tgt}")

        # with open(self.jsonl_path, "r") as f:
        #     self.scene_paths = [line.strip() for line in f]

        
    def load_eval_config(self, config_path: str) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """
        Load evaluation configuration from JSON file.
        
        Returns:
            test_paths, test_indices_src, test_indices_tgt
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, "r") as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ["test_paths", "test_indices_src", "test_indices_tgt"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Resolve relative paths to absolute (optional but recommended)
        config_dir = path.parent
        test_paths = [
            str((config_dir / p).resolve()) if not Path(p).is_absolute() else p
            for p in config["test_paths"]
        ]
        
        return test_paths, config["test_indices_src"], config["test_indices_tgt"]


    def read_scene(self, scene_path, chosen_indices):
        
        scene_path = scene_path.strip()
        if self.hdf5_path is not None:
            hdf5_path = self.hdf5_path
        else:
            hdf5_path = glob.glob(osp.join(self.meta_folder, '*.hdf5'))[0]
        
        # pass
        with h5py.File(hdf5_path, 'r') as f:
            if scene_path not in f:
                raise KeyError(f"Sequence '{scene_path}' not found in {hdf5_path}")
            
            grp = f[scene_path]
            n = grp['K'].shape[0]

            scene_samples = []
            for idx in chosen_indices:
                # image_path = os.path.join(scene_path, grp['image_path'][idx].decode('utf-8')).replace("data","data1", 1)
                image_path = os.path.join(scene_path, grp['image_path'][idx].decode('utf-8'))
                # print(f"image_path: {image_path}")
                d = {
                    'sequence_name': scene_path,
                    'frame_number': int(grp['frame_number'][idx, 0]),
                    'image_path': image_path,
                    'R': grp['R'][idx].tolist(),
                    'T': grp['T'][idx].tolist(),
                    'K': grp['K'][idx].tolist(),
                    'text': grp['text'][idx].decode('utf-8') if 'text' in grp else "",
                }
                scene_samples.append(d)

        return scene_samples

    def prepare_model_input_multiview(self, scene_samples):
        h_div_w = 1.0
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]

        scene_text = []
        scene_img = []

        scene_poses = []
        scene_intrs = []

        for i in range(len(scene_samples)):
            img_path = scene_samples[i]['image_path']
            short_text_input = scene_samples[i]['text']
            text_input = short_text_input

            scene_text.append(text_input)
            with open(img_path, 'rb') as f:
                try:
                    img: PImage.Image = PImage.open(f)
                    img = img.convert("RGBA")
                    background = PImage.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img=background
                    # img = img.convert('RGB')
                except:
                    print(f"exception for image: {img_path}")
                    return False, None
                tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
                K = torch.tensor(scene_samples[i]['K'])

                img_B3HW, new_K = transform_wintr(img, K, tgt_h, tgt_w)
                scene_img.append(img_B3HW)

                # create plucker raymap corresponding to image
                R = torch.tensor(scene_samples[i]['R'])
                T = torch.tensor(scene_samples[i]['T'])

                pose = torch.eye(4)
                pose[:3,:3] = R
                pose[:3,3] = T


                scene_poses.append(pose)
                scene_intrs.append(new_K)


        return True, (scene_text, torch.stack(scene_img), torch.stack(scene_poses), torch.stack(scene_intrs))

    def normalize_scene_batches(self, src_samples, tgt_samples):
        """
        Normalizes source and target views into a shared coordinate system.
        1. Centers the scene on the first source view (ref0 -> Identity).
        2. Scales the scene so the furthest source view is at distance 1.0.
           (Safeguard against explosion).
        """
        # 1. Combine all samples to process tensors efficiently
        all_samples = src_samples + tgt_samples
        if len(all_samples) == 0:
            return

        # 2. Extract C2W matrices
        # Assuming data['R'] and data['T'] form the C2W matrix directly
        c2ws = []
        for s in all_samples:
            pose = torch.eye(4)
            pose[:3, :3] = torch.tensor(s['R'])
            pose[:3, 3] = torch.tensor(s['T'])
            c2ws.append(pose)
        c2ws = torch.stack(c2ws) # [N, 4, 4]

        # 3. Define Reference Transform (World -> Ref0)
        # We transform everything so src_samples[0] becomes the origin.
        ref_c2w = c2ws[0] 
        ref_w2c = torch.linalg.inv(ref_c2w)

        # Apply transformation: New_Pose = Ref_W2C @ Old_Pose
        c2ws_centered = torch.matmul(ref_w2c.unsqueeze(0), c2ws)

        # 4. Determine Scale Factor
        # Instead of using specifically src[1] (which might be 1cm away),
        # we find the source view furthest from the origin and map THAT to 1.0.
        num_src = len(src_samples)
        src_c2ws = c2ws_centered[:num_src]
        
        # Calculate distances of all source cameras from the new origin
        # dists = torch.norm(src_c2ws[:, :3, 3], dim=-1)
        dists = torch.norm(c2ws_centered[:, :3, 3], dim=-1)

        max_dist = dists.max()
        
        # Safer Style (Max Radius):
        # Ensures no source camera coordinate > 1.0, preventing PRoPE NaNs.
        if max_dist > 1e-5:
            scale = 1.0 / max_dist
        else:
            scale = 1.0 # Fallback for single-view or stationary camera

        # 5. Apply Scale to Translation
        c2ws_centered[:, :3, 3] *= scale

        # 6. Write back to dictionaries
        for i, s in enumerate(all_samples):
            # Update the dictionary in-place
            new_pose = c2ws_centered[i]
            s['R'] = new_pose[:3, :3].tolist()
            s['T'] = new_pose[:3, 3].tolist()


class MultiviewIterableDataset(IterableDataset):
    def __init__(
        self, 
        meta_folder: str, 
        max_caption_len=512, 
        short_prob=0.2, 
        load_vae_instead_of_image=False,
        buffersize: int = 10000,
        seed: int = 0, 
        pn: str = '',
        online_t5: bool = True,
        batch_size: int = 2,
        num_replicas: int = 1, # 1,
        rank: int = 0, # 0
        dataloader_workers: int = 2,
        dynamic_resolution_across_gpus: bool = True,
        enable_dynamic_length_prompt: bool = True,
        N_views_src=0,
        N_views_tgt=0,
        N_views_total=0,
        identity_test=False,
        hdf5_path=None,
        **kwargs,
    ):

        self.meta_folder = meta_folder
        self.pn = pn
        self.online_t5 = online_t5
        self.buffer_size = buffersize
        self.num_replicas = num_replicas
        self.rank = rank
        self.worker_id = 0
        self.global_worker_id = 0
        self.dataloader_workers = max(1, dataloader_workers)
        self.max_caption_len = max_caption_len
        self.short_prob = short_prob
        self.load_vae_instead_of_image = load_vae_instead_of_image # set to false
        self.dynamic_resolution_across_gpus = dynamic_resolution_across_gpus
        self.enable_dynamic_length_prompt = enable_dynamic_length_prompt
        self.batch_size = batch_size
        # print(f'self.dynamic_resolution_across_gpus: {self.dynamic_resolution_across_gpus}')
        # print(f'self.enable_dynamic_length_prompt: {self.enable_dynamic_length_prompt}')
        # print(f'self.buffer_size: {self.buffer_size}')
        self.shuffle = True
        self.global_workers = self.num_replicas * self.dataloader_workers
        self.h_div_w_template2generator, self.samples_div_gpus_workers_batchsize_2batches, total_samples = self.set_h_div_w_template2generator()
        self.split_meta_files()
        self.seed = seed
        self.epoch_worker_generator = None
        self.epoch_global_worker_generator = None
        self.set_epoch(0)
        print(f'num_replicas: {num_replicas}, rank: {rank}, dataloader_workers: {dataloader_workers}, seed:{seed}, samples_div_gpus_workers_batchsize_2batches: {self.samples_div_gpus_workers_batchsize_2batches}')

        assert N_views_total==0 or N_views_src==N_views_tgt==0

        self.N_views_src = N_views_src
        self.N_views_tgt = N_views_tgt
        self.N_views_total = N_views_total

        self.identity_test = identity_test
        self.hdf5_path = hdf5_path

    def set_h_div_w_template2generator(self,):
        samples_div_gpus_workers_batchsize_2batches = 0
        h_div_w_template2generator = {}
        total_samples = 0

        # go over all jsonl files(aspect ratios)
        for filepath in sorted(glob.glob(osp.join(self.meta_folder, '*.jsonl'))):
            filename = osp.basename(filepath)
            h_div_w_template, num_of_samples = osp.splitext(filename)[0].split('_')
            total_samples += int(num_of_samples)

        for filepath in sorted(glob.glob(osp.join(self.meta_folder, '*.jsonl'))):
            filename = osp.basename(filepath)
            h_div_w_template, num_of_samples = osp.splitext(filename)[0].split('_')
            num_of_samples = int(num_of_samples)
            if num_of_samples < self.global_workers:
                print(f'{filepath} has too few examples ({num_of_samples}, proportion: {num_of_samples/total_samples*100:.1f}%), < global workers ({self.global_workers})! Skip h_div_w_template: {h_div_w_template}')
                continue
            print(f'{filepath} has sufficient examples ({num_of_samples}), proportion: {num_of_samples/total_samples*100:.1f}%, > global workers ({self.global_workers})! Preserve h_div_w_template: {h_div_w_template}')
            num_of_batches = max(1, int((num_of_samples // self.global_workers // self.batch_size)))
            h_div_w_template2generator[h_div_w_template] = {
                'filepath': filepath,
                'num_of_samples': num_of_samples,
                'num_of_batches': num_of_batches,
            }
            samples_div_gpus_workers_batchsize_2batches += num_of_batches
        return h_div_w_template2generator, samples_div_gpus_workers_batchsize_2batches, total_samples

    def split_meta_files(self, ):
        print('[data preprocess] split_meta_files')

        def split_and_sleep(generator_info):
            missing, chunk_id2save_files = get_part_jsonls(generator_info['filepath'], generator_info['num_of_samples'], parts=self.num_replicas)
            if missing:
                tdist.barrier()
                if self.rank == 0:
                    split_large_txt_files(generator_info['filepath'], chunk_id2save_files)
                else:
                    sleep_time = int(generator_info['num_of_samples'] / 30000000 * 10)
                    print(f'[data preprocess] sleep {sleep_time} minutes awaiting rank0 split_meta_files...')
                    time.sleep(sleep_time*60)
                tdist.barrier()
            generator_info['part_filepaths'] = sorted(list(chunk_id2save_files.values()))
            return generator_info

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(split_and_sleep, generator_info): h_div_w_template for h_div_w_template, generator_info in self.h_div_w_template2generator.items()}
            for future in concurrent.futures.as_completed(futures):
                h_div_w_template = futures[future]
                self.h_div_w_template2generator[h_div_w_template] = future.result()
                # try:
                #     self.h_div_w_template2generator[h_div_w_template] = future.result()
                # except Exception as exc:
                #     print(f'[data preprocess] h_div_w_template {h_div_w_template} generated an exception: {exc}')

        print('[data preprocess] split_meta_files done')

    def set_global_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1
        assert worker_total_num == self.dataloader_workers, print(worker_total_num, self.dataloader_workers)
        self.worker_id = worker_id
        self.global_worker_id = self.rank * self.dataloader_workers + worker_id
        # print(f'Set worker_id to {self.worker_id}, global_worker_id to {self.global_worker_id}')
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.set_generator()
    
    def set_generator(self, ):
        self.epoch_worker_generator = np.random.default_rng(self.seed + self.epoch + self.worker_id)
        self.epoch_global_worker_generator = np.random.default_rng(self.seed + self.epoch + self.global_worker_id)
    
    def get_h_div_w_template_2_unlearned_batches(self,):
        h_div_w_template_2_unlearned_batches = {}
        total_unlearned_batches = 0
        for h_div_w_template, generator_info in self.h_div_w_template2generator.items():
            h_div_w_template_2_unlearned_batches[h_div_w_template] = generator_info['num_of_batches']
            total_unlearned_batches += generator_info['num_of_batches']
        self.total_unlearned_batches = total_unlearned_batches
        self.h_div_w_template_2_unlearned_batches = h_div_w_template_2_unlearned_batches
        assert self.total_unlearned_batches == self.samples_div_gpus_workers_batchsize_2batches

    def _next_h_div_w_template(self,):
        while True:
            self.get_h_div_w_template_2_unlearned_batches()
            while self.total_unlearned_batches > 0:
                if self.dynamic_resolution_across_gpus:
                    i = self.epoch_global_worker_generator.integers(0, self.total_unlearned_batches)
                else:
                    i = self.epoch_worker_generator.integers(0, self.total_unlearned_batches)
                self.total_unlearned_batches -= 1
                for h_div_w_template, unlearned_batches in self.h_div_w_template_2_unlearned_batches.items():
                    if i < unlearned_batches:
                        yield h_div_w_template
                        self.h_div_w_template_2_unlearned_batches[h_div_w_template] -= 1
                        break
                    else:
                        i -= unlearned_batches

    def read_scene(self, scene_path, N_views=3, indices=None):
        
        # print(f"scene_path[-2:]: {scene_path[-2:]}")
        # if scene_path[-2:] == '\n':
        #     scene_path = scene_path[:-2]
        scene_path = scene_path.strip()

        if self.hdf5_path is not None:
            hdf5_path = self.hdf5_path
        else:
            hdf5_path = glob.glob(osp.join(self.meta_folder, '*.hdf5'))[0]
        
        # pass
        with h5py.File(hdf5_path, 'r') as f:
            if scene_path not in f:
                raise KeyError(f"Sequence '{scene_path}' not found in {hdf5_path}")
            
            grp = f[scene_path]
            n = grp['K'].shape[0]
    
            if indices is not None:
                chosen_indices = indices
            else:
                if n >= N_views:
                    chosen_indices = random.sample(range(n), N_views)
                else:
                    # allow repeats when not enough frames
                    chosen_indices = random.choices(range(n), k=N_views)

            # try:
            #     chosen_indices = random.sample(range(n), n_views)
            # except:
            #     print(f"scene_path: {scene_path} | n: {n} | n_views: {n_views}")
            #     raise

            scene_samples = []
            for idx in chosen_indices:
                # image_path = os.path.join(scene_path, grp['image_path'][idx].decode('utf-8')).replace("data","data1", 1)
                image_path = os.path.join(scene_path, grp['image_path'][idx].decode('utf-8'))
                # print(f"image_path: {image_path}")
                d = {
                    'sequence_name': scene_path,
                    'frame_number': int(grp['frame_number'][idx, 0]),
                    'image_path': image_path,
                    'R': grp['R'][idx].tolist(),
                    'T': grp['T'][idx].tolist(),
                    'K': grp['K'][idx].tolist(),
                    'text': grp['text'][idx].decode('utf-8') if 'text' in grp else "",
                }
                scene_samples.append(d)

            # print(f"scene_samples: {scene_samples}")

    
            # print(f"n: {n} chosen_indices: {chosen_indices}")

        return scene_samples

    def __iter__(self):
        self.set_global_worker_id()
        self.set_generator()
        
        # print(f"self.h_div_w_template2generator.items(): {self.h_div_w_template2generator.items()}")

        # dict_items([('1.000', {'filepath': 'data/co3d_data/1.000_000025000.jsonl', 
        # 'num_of_samples': 25000, 
        # 'num_of_batches': 6250, 
        # 'part_filepaths': ['data/co3d_data/1.000_000025000.jsonl']})])

        # dict_items([('1.000', {'filepath': 'data/infinity_toy_data/splits/1.000_000002500.jsonl', 
        # 'num_of_samples': 2500, 
        # 'num_of_batches': 625, 
        # 'part_filepaths': ['data/infinity_toy_data/splits/1.000_000002500.jsonl']}), 
        # ('1.500', {'filepath': 'data/infinity_toy_data/splits/1.500_000002500.jsonl', 
        # 'num_of_samples': 2500, 
        # 'num_of_batches': 625, 
        # 'part_filepaths': ['data/infinity_toy_data/splits/1.500_000002500.jsonl']})])

        for h_div_w_template, generator_info in self.h_div_w_template2generator.items():
            # print(f"h_div_w_template: {h_div_w_template}")
            proportion = generator_info['num_of_batches'] / self.samples_div_gpus_workers_batchsize_2batches
            h_div_w_buffer_size = int(self.buffer_size * proportion)
            h_div_w_buffer_size = min(max(1, h_div_w_buffer_size), generator_info['num_of_batches'] * self.batch_size)
            # print(f"h_div_w_buffer_size: {h_div_w_buffer_size}")
            if 'mem_buffer' in generator_info:
                del generator_info['mem_buffer']
            mem_buffer = []
            for _ in range(h_div_w_buffer_size):
                mem_buffer.append(self.infinite_next(generator_info))
            # print(f"[0] len(mem_buffer): {len(mem_buffer)}")
            generator_info['mem_buffer'] = mem_buffer
        
        
        # print(f"generator_info: {generator_info}")
        next_h_div_w_template_iter = self._next_h_div_w_template()
        # print(f"self.samples_div_gpus_workers_batchsize_2batches: {self.samples_div_gpus_workers_batchsize_2batches}")
        # while True:
        for _ in range(self.samples_div_gpus_workers_batchsize_2batches):
            batch_data_src = []
            batch_data_tgt = []
            h_div_w_template = next(next_h_div_w_template_iter)

            if self.N_views_src>0 and self.N_views_tgt>0:
                N_views_src = self.N_views_src
                N_views_tgt = self.N_views_tgt
            else:
                # non-fixed number of source/target views
                # src: min=1, max=self.N_views_total
                N_views_src = random.randint(1, self.N_views_total-1)
                N_views_tgt = self.N_views_total - N_views_src


            while len(batch_data_src) < self.batch_size:
                # try:
                generator_info = self.h_div_w_template2generator[h_div_w_template]
                # print(f"Multiview::__iter__ generator_info: {generator_info}")
                mem_buffer = generator_info['mem_buffer']
                i = self.epoch_global_worker_generator.integers(0, len(mem_buffer))
                data_item = mem_buffer[i]
                # print(f"data_item: {data_item}")
                
                mem_buffer[i] = self.infinite_next(generator_info)
                
                scene_sample_src = self.read_scene(data_item, N_views_src)
                scene_sample_tgt = self.read_scene(data_item, N_views_tgt)

                self.normalize_scene_batches(scene_sample_src, scene_sample_tgt)
                
                # ret, model_input = self.prepare_model_input(json.loads(data_item)) # data_item[0] is row number of panda dataframe
                # print(f"scene_sample: {scene_sample}")
                ret1, model_input_src = self.prepare_model_input_multiview(scene_sample_src)
                ret2, model_input_tgt = self.prepare_model_input_multiview(scene_sample_tgt)
                # print(f"model_input[1].shape: {model_input[1].shape}")
                if ret1 and ret2:
                    # c_, h_, w_ = model_input[1].shape[-3:]
                    #TODO maybe return??
                    # if c_ != 3 or np.abs(h_/w_-float(h_div_w_template)) > 0.01:
                    # if c_ != 3:
                    #     print(f'Croupt data item: {data_item}')
                    # else:
                    #     batch_data.append(model_input)
                    batch_data_src.append(model_input_src)
                    batch_data_tgt.append(model_input_tgt)
                else:
                    continue
                del data_item
                # except Exception as e:
                #     print(e)


            captions_src = [item[0] for item in batch_data_src]
            images_src = torch.stack([item[1] for item in batch_data_src])
            poses_src = torch.stack([item[2] for item in batch_data_src])
            intrs_src = torch.stack([item[3] for item in batch_data_src])

            captions_tgt = [item[0] for item in batch_data_tgt]
            images_tgt = torch.stack([item[1] for item in batch_data_tgt])
            poses_tgt = torch.stack([item[2] for item in batch_data_tgt])
            intrs_tgt = torch.stack([item[3] for item in batch_data_tgt])

            yield (images_src, captions_src, poses_src, intrs_src, images_tgt, captions_tgt, poses_tgt, intrs_tgt)
            
            del batch_data_src
            del images_src
            del captions_src
            del poses_src
            del intrs_src

            del batch_data_tgt
            del images_tgt
            del captions_tgt
            del poses_tgt
            del intrs_tgt
    
    def infinite_next(self, generator_info):
        # print(f"generator_info.keys(): {generator_info.keys()}")
        # print(f"generator_info['sub_iterator']: {generator_info['sub_iterator']}")
        try:
            if 'sub_iterator' not in generator_info:
                raise StopIteration
            return next(generator_info['sub_iterator'])
        except StopIteration as e:
            # print(f"except")
            if 'record_iterator' in generator_info:
                generator_info['record_iterator'].close()
            if 'sub_iterator' in generator_info:
                del generator_info['sub_iterator']
            part_filepath = generator_info['part_filepaths'][self.rank]
            # print(f"part_filepath: {part_filepath}")
            generator_info['record_iterator'] = open(part_filepath, 'r')
            part_num_of_samples = int(osp.splitext(osp.basename(part_filepath))[0].split('_')[-1])
            # print(f'part_filepath: {part_filepath}, rank: {self.rank}, worker_id:{self.worker_id}, part_num_of_samples: {part_num_of_samples}, dataloader_workers: {self.dataloader_workers}')
            generator_info['sub_iterator'] = itertools.islice(generator_info['record_iterator'], self.worker_id, part_num_of_samples, self.dataloader_workers)
            return next(generator_info['sub_iterator'])

    def __len__(self):
        return self.samples_div_gpus_workers_batchsize_2batches * self.dataloader_workers
    
    def total_samples(self):
        return self.samples_div_gpus_workers_batchsize_2batches * self.dataloader_workers * self.num_replicas * self.batch_size

    def get_text_input(self, long_text_input, short_text_input, long_text_type):
        random_value = self.epoch_global_worker_generator.random()
        if self.enable_dynamic_length_prompt and long_text_type != 'user_prompt':
            long_text_elems = [item for item in long_text_input.split('.') if item]
            if len(long_text_elems):
                first_sentence_words = [item for item in long_text_elems[0].split(' ') if item]
            else:
                first_sentence_words = 0
            if len(first_sentence_words) >= 15:
                num_sentence4short_text = 1
            else:
                num_sentence4short_text = 2
            if not short_text_input:
                short_text_input = '.'.join(long_text_elems[:num_sentence4short_text])
            if random_value < self.short_prob:
                return short_text_input
            if len(long_text_elems) <= num_sentence4short_text:
                return long_text_input
            select_sentence_num = self.epoch_global_worker_generator.integers(num_sentence4short_text+1, len(long_text_elems)+1)
            return '.'.join(long_text_elems[:select_sentence_num])
        else:
            if short_text_input and random_value < self.short_prob:
                return short_text_input
            return long_text_input

    def normalize_scene_batches(self, src_samples, tgt_samples):
        """
        Normalizes source and target views into a shared coordinate system.
        1. Centers the scene on the first source view (ref0 -> Identity).
        2. Scales the scene so the furthest source view is at distance 1.0.
           (Safeguard against explosion).
        """
        # 1. Combine all samples to process tensors efficiently
        all_samples = src_samples + tgt_samples
        if len(all_samples) == 0:
            return

        # 2. Extract C2W matrices
        # Assuming data['R'] and data['T'] form the C2W matrix directly
        c2ws = []
        for s in all_samples:
            pose = torch.eye(4)
            pose[:3, :3] = torch.tensor(s['R'])
            pose[:3, 3] = torch.tensor(s['T'])
            c2ws.append(pose)
        c2ws = torch.stack(c2ws) # [N, 4, 4]

        # 3. Define Reference Transform (World -> Ref0)
        # We transform everything so src_samples[0] becomes the origin.
        ref_c2w = c2ws[0] 
        ref_w2c = torch.linalg.inv(ref_c2w)

        # Apply transformation: New_Pose = Ref_W2C @ Old_Pose
        c2ws_centered = torch.matmul(ref_w2c.unsqueeze(0), c2ws)

        # 4. Determine Scale Factor
        # Instead of using specifically src[1] (which might be 1cm away),
        # we find the source view furthest from the origin and map THAT to 1.0.
        num_src = len(src_samples)
        src_c2ws = c2ws_centered[:num_src]
        
        # Calculate distances of all source cameras from the new origin
        # dists = torch.norm(src_c2ws[:, :3, 3], dim=-1)
        dists = torch.norm(c2ws_centered[:, :3, 3], dim=-1)

        max_dist = dists.max()
        
        # Safer Style (Max Radius):
        # Ensures no source camera coordinate > 1.0, preventing PRoPE NaNs.
        if max_dist > 1e-5:
            scale = 1.0 / max_dist
        else:
            scale = 1.0 # Fallback for single-view or stationary camera

        # 5. Apply Scale to Translation
        c2ws_centered[:, :3, 3] *= scale

        # 6. Write back to dictionaries
        for i, s in enumerate(all_samples):
            # Update the dictionary in-place
            new_pose = c2ws_centered[i]
            s['R'] = new_pose[:3, :3].tolist()
            s['T'] = new_pose[:3, 3].tolist()

    def prepare_model_input(self, data_item) -> Tuple:
        """
        data_item keys:
        1. image_path
        2. h_div_w
        3. text
        4. long_caption
        5. long_caption_type
        """

        # print(f"data_item: {data_item}")
        img_path, h_div_w = data_item['image_path'], data_item['h_div_w']
        short_text_input, long_text_input = data_item['text'], data_item['long_caption']
        long_text_type = data_item.get('long_caption_type', 'user_prompt')
        # print(f"img_path: {img_path}")
        # print(f"short_text_input: {short_text_input}")
        # print(f"long_text_input: {long_text_input}")
        # print(f"long_text_type: {long_text_type}")
        text_input = self.get_text_input(long_text_input, short_text_input, long_text_type)
        # print(f"text_input: {text_input}")
        text_input = process_short_text(text_input)

        h_div_w = 1.0
        

        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        print(f"h_div_w_template: {h_div_w_template}")
        try:
            if self.load_vae_instead_of_image:
                img_B3HW = None
                vae_path = self.get_vae_path(img_path)
                with open(vae_path, 'rb') as f:
                    gt_ms_idx_Bl = pickle.load(f)
            else:
                gt_ms_idx_Bl = None
                with open(img_path, 'rb') as f:
                    img: PImage.Image = PImage.open(f)
                    img = img.convert('RGB')
                    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
                    print(f"tgt_h: {tgt_h} | tgt_w: {tgt_w}")
                    print(f"img.size: {img.size}")
                    img_B3HW = transform(img, tgt_h, tgt_w)
                    torchvision.utils.save_image(img_B3HW, img_path.split("/")[-1])
                    print(f"dtype(img_B3HW): {type(img_B3HW)}")
                    print(f"img_B3HW.shape: {img_B3HW.shape}")
            if not self.online_t5:
                short_t5_path, long_t5_path = self.get_t5_path(img_path)
                if self.epoch_global_worker_generator.random() <= self.short_prob:
                    t5_path = short_t5_path
                else:
                    t5_path = long_t5_path
                t5_meta = np.load(t5_path)
                text_input = t5_meta['t5_feat'][:self.max_caption_len] # L x C
        except Exception as e:
            print(f'input error: {e}, skip to another index')
            return False, None

        if self.load_vae_instead_of_image:
            return True, (text_input, *gt_ms_idx_Bl)
        else:
            return True, (text_input, img_B3HW)


    def prepare_model_input_multiview(self, scene_samples):
        h_div_w = 1.0
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]

        scene_text = []
        scene_img = []

        scene_poses = []
        scene_intrs = []

        for i in range(len(scene_samples)):
            img_path = scene_samples[i]['image_path']
            short_text_input = scene_samples[i]['text']
            text_input = short_text_input

            scene_text.append(text_input)
            with open(img_path, 'rb') as f:
                try:
                    img: PImage.Image = PImage.open(f)
                    img = img.convert("RGBA")
                    background = PImage.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img=background
                    # img = img.convert('RGB')
                except:
                    print(f"exception for image: {img_path}")
                    return False, None
                tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
                K = torch.tensor(scene_samples[i]['K'])

                img_B3HW, new_K = transform_wintr(img, K, tgt_h, tgt_w)
                scene_img.append(img_B3HW)

                # create plucker raymap corresponding to image
                R = torch.tensor(scene_samples[i]['R'])
                T = torch.tensor(scene_samples[i]['T'])

                pose = torch.eye(4)
                pose[:3,:3] = R
                pose[:3,3] = T


                scene_poses.append(pose)
                scene_intrs.append(new_K)


        return True, (scene_text, torch.stack(scene_img), torch.stack(scene_poses), torch.stack(scene_intrs))








    @staticmethod
    def collate_function(batch, online_t5: bool = False) -> None:
        pass

if __name__ == '__main__':
    # torchrun --nnodes=1 --nproc-per-node=2 --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT dataset/dataset_t2i_iterable.py
    tdist.init_process_group(backend='nccl')
    batch_size = 2
    dataloader_workers = 12
    dataset = Multiview(
        args=None, 
        meta_folder='data/train_splits/xxx_pretrain/jsonl_files_filter_duplicate_captions',
        data_load_reso=None, 
        max_caption_len=512, 
        short_prob=1.0, 
        load_vae_instead_of_image=False,
        buffersize=100000,
        seed=0, 
        online_t5=True,
        pn='0.06M',
        batch_size=batch_size,
        num_replicas=8, # tdist.get_world_size(),
        rank=tdist.get_rank(), # 0
        dataloader_workers=dataloader_workers,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=dataloader_workers)
    print(f'len(dataloader): {len(dataloader)}, len(dataset): {len(dataset)}, total_samples: {dataset.total_samples()}')
    t1 = time.time()
    h_div_w2samples = {}
    for ep in range(4):
        dataloader.dataset.set_epoch(ep)
        pbar = tqdm.tqdm(total=len(dataloader))
        for i, data in enumerate(iter(dataloader)):
            pbar.update(1)
            t2 = time.time()
            h_div_w = data[0].shape[-2] / data[0].shape[-1]
            h_div_w = f'{h_div_w:.3f}'
            if h_div_w not in h_div_w2samples:
                h_div_w2samples[h_div_w] = 0
            h_div_w2samples[h_div_w] += 1
            if (i+1) % 100 == 0:
                total_samples = np.sum(list(h_div_w2samples.values()))
                print()
                for h_div_w, num in sorted(h_div_w2samples.items()):
                    print(f'h_div_w: {h_div_w}, samples: {num}, proportion: {num/total_samples*100:.1f}%')
                print()
            t1 = time.time()
