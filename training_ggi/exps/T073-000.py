# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Any, Dict
from torchvision.transforms import Resize
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from controlnet_aux import HEDdetector

import diffusers
import torch
import transformers
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from accelerate import cpu_offload
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel
import numpy as np
import rp

import sys

sys.path.append('.')

from noise_warp import NoiseWarper
from raft import RaftOpticalFlow
# from myvideodataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip
from mydataset_depthhed_mask import MultiviewDataset, CollateFunction
from text_encoder import compute_prompt_embeddings  # isort:skip
from myutils import ch_channel_cnn
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)


def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1 - alpha)) / (alpha**2 +
                                                                          (1 - alpha)**2)**.5


def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise), alpha)
    elif isinstance(noise, np.ndarray):
        return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else:
        raise TypeError(
            f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")


def downscale_noise(noise, downscale_factor=32):
    down_noise = rp.torch_resize_image(noise, 1 / downscale_factor, interp='area')
    down_noise = down_noise * downscale_factor
    return down_noise


logger = get_logger(__name__)


def main(args):

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                      logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl",
                                                       timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower(
    ) else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    # CHANGE CHANNEL ---------------------------------------------------------------------------- #
    in_channels = 48  # noise 16 + image 16 + LSD 16
    ch_channel_cnn(transformer, in_channels)
    transformer._internal_dict['in_channels'] = in_channels
    # ------------------------------------------------------------------------------------------- #

    # Hack (LoRA Init) -------------------------------------------------------------------------- #
    transformer_lora_config = LoraConfig(
        r=2048,
        lora_alpha=2048,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    lora_state_dict = load_file('lora_models/I2V5B_final_i38800_nearest_lora_weights.safetensors')
    transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v
        for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    incompatible_keys = set_peft_model_state_dict(
        transformer,
        transformer_state_dict,
        adapter_name="default",
    )
    # ------------------------------------------------------------------------------------------- #

    if args.ignore_learned_positional_embeddings:
        del transformer.patch_embed.pos_embedding
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                                      subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(True)

    # lora wieght fix ---------------------------------------------------------------------------- #
    for name, p in transformer.named_parameters():
        if 'lora' in name:
            p.requires_grad_(False)
    # -------------------------------------------------------------------------------------------- #

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2**(len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if ("fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
                and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]):
            weight_dtype = torch.float16
        if ("bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
                and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if gethostname() == "comar-System-Product-Name":
        vae.to(dtype=weight_dtype)
        cpu_offload(vae, accelerator.device)
        text_encoder.to(dtype=weight_dtype)
        cpu_offload(text_encoder, accelerator.device)
        transformer.to(accelerator.device, dtype=weight_dtype)
    else:
        text_encoder.to(dtype=weight_dtype)
        cpu_offload(text_encoder, accelerator.device)
        # text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(dtype=weight_dtype)
        cpu_offload(vae, accelerator.device)
        transformer.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(accelerator, model),
                              type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    model.save_pretrained(os.path.join(output_dir, "transformer"),
                                          safe_serialization=True,
                                          max_shard_size="5GB")
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model),
                              type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(
                        f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModel.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer")
                init_under_meta = True

        # load_model = CogVideoXTransformer3DModel.from_pretrained(os.path.join(input_dir, "transformer"))
        # load_model = CogVideoXTransformer3DModel.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     subfolder="transformer",
        #     torch_dtype=load_dtype,
        #     revision=args.revision,
        #     variant=args.variant,
        # )
        # transformer_lora_config = LoraConfig(
        #     r=2048,
        #     lora_alpha=2048,
        #     init_lora_weights=True,
        #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        # )
        # load_model.add_adapter(transformer_lora_config)
        config = OmegaConf.load(os.path.join(input_dir, "transformer", 'config.json'))
        transformer_.register_to_config(**config)

        # from safetensors.torch import load_file
        shard_files = glob(os.path.join(input_dir, "transformer", '*.safetensors'))
        state_dict = {}
        for shard_file in shard_files:
            tensors = load_file(shard_file)
            state_dict.update(tensors)
        _ = transformer_.load_state_dict(state_dict, assign=init_under_meta)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps *
                              args.train_batch_size * accelerator.num_processes)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize
                                   for param in model["params"])

    use_deepspeed_optimizer = (accelerator.state.deepspeed_plugin is not None and "optimizer"
                               in accelerator.state.deepspeed_plugin.deepspeed_config)
    use_deepspeed_scheduler = (accelerator.state.deepspeed_plugin is not None and "scheduler"
                               in accelerator.state.deepspeed_plugin.deepspeed_config)

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    # Dataset and DataLoader -------------------------------------------------------------------- #
    collate_fn = CollateFunction(weight_dtype, )
    train_dataset = MultiviewDataset(
        path_root=args.data_root,
        height=args.height,
        width=args.width,
        n_sample_frames=args.max_num_frames - 3,
        frame_intervals=args.frame_intervals,
        device=accelerator.device,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        # num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
    )
    # ------------------------------------------------------------------------------------------- #

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer,
                                                                   train_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-sft"
        accelerator.init_trackers(tracker_name)

        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if gethostname() == "comar-System-Product-Name":
        args.resume_from_checkpoint = None

    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer,
                                                        "module") else transformer.config
    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    # HACK -------------------------------------------------------------------------------------- #
    FRAME = 2**-1
    FLOW = 2**3
    LATENT = 8
    noise_channels = 16
    resize_flow = FLOW
    downscale = round(FRAME * FLOW) * LATENT
    raft_model = RaftOpticalFlow(accelerator.device, "large").model
    # ------------------------------------------------------------------------------------------- #

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            logs = {}

            with accelerator.accumulate(models_to_accumulate):
                # images = batch["images"].to(accelerator.device, non_blocking=True)
                # images_last = batch["images_last"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                conds = batch["conds"]
                # masks = batch["masks"]

                videos = torch.cat([videos, videos[:, -1:], videos[:, -1:], videos[:, -1:]], dim=1)
                conds = torch.cat([conds, conds[:, -1:], conds[:, -1:], conds[:, -1:]], dim=1)
                # masks = torch.cat([masks, masks[:, -1:], masks[:, -1:], masks[:, -1:]], dim=1)

                # Generate noise ---------------------------------------------------------------- #
                bs, _, _, h, w = videos.shape
                h = h // 2
                w = w // 2
                noisess = []
                for b in range(bs):
                    warper = NoiseWarper(
                        c=noise_channels,
                        h=resize_flow * h,
                        w=resize_flow * w,
                        device=accelerator.device,
                        post_noise_alpha=0,
                        progressive_noise_alpha=0,
                        warp_kwargs={},
                    )
                    noises = [downscale_noise(warper.noise, downscale)]
                    for prev_video_frame, video_frame in list(zip(videos[b], videos[b][1:])):

                        prev_video_frame = prev_video_frame.to(dtype=torch.float32)
                        video_frame = video_frame.to(dtype=torch.float32)
                        prev_video_frame = Resize((h, w))(prev_video_frame)[None]
                        video_frame = Resize((h, w))(video_frame)[None]

                        dx, dy = raft_model(prev_video_frame, video_frame)[-1][0]
                        noise = warper(dx, dy).noise
                        down_noise = downscale_noise(noise, downscale)
                        noises.append(down_noise)
                    noises = torch.stack(noises).to(dtype=weight_dtype)
                    noises = rp.resize_list(noises, (args.max_num_frames - 1) // 4 + 1)
                    noisess.append(noises)
                noise = torch.stack(noisess)
                alpha = random.random()
                noise = mix_new_noise(noise, alpha)
                torch.cuda.empty_cache()
                # ------------------------------------------------------------------------------- #

                # Encode Images
                images = videos[:, [0, -1]]
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                image_noise_sigma = torch.normal(mean=-3.0,
                                                 std=0.5,
                                                 size=(images.size(0), ),
                                                 device=accelerator.device,
                                                 dtype=weight_dtype)
                image_noise_sigma = torch.exp(image_noise_sigma)
                noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None,
                                                                                     None, None]
                noisy_images_first, noisy_images_end = noisy_images[:, :, [0]], noisy_images[:, :,
                                                                                             [1]]

                image_latent_dist = vae.encode(noisy_images_first).latent_dist
                image_end_latent_dist = vae.encode(noisy_images_end).latent_dist

                image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format,
                                                 dtype=weight_dtype)

                image_end_latents = image_end_latent_dist.sample() * VAE_SCALING_FACTOR
                image_end_latents = image_end_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                image_end_latents = image_end_latents.to(memory_format=torch.contiguous_format,
                                                         dtype=weight_dtype)
                # Masking -----------------------------------------------------------#
                # masks = masks.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                # masks = masks[:, [-1], :1].permute(0, 2, 1, 3, 4)
                # masks = masks[..., ::8, ::8]
                # image_end_latents = image_end_latents * masks
                # -------------------------------------------------------------------#

                # Encode videos
                videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                latent_dist = vae.encode(videos).latent_dist

                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format,
                                                 dtype=weight_dtype)

                # Encode conditions
                conds = conds.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                latent_dist = vae.encode(conds).latent_dist

                cond_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                cond_latents = cond_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                cond_latents = cond_latents.to(memory_format=torch.contiguous_format,
                                               dtype=weight_dtype)

                # Encode videos
                padding_shape = (video_latents.shape[0], video_latents.shape[1] - 2,
                                 *video_latents.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding, image_end_latents], dim=1)

                # if random.random() < args.noised_image_dropout:
                #     image_latents = torch.zeros_like(image_latents)

                # Encode prompts
                if not args.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise that will be added to the latents
                # noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size, ),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (prepare_rotary_positional_embeddings(
                    height=height * VAE_SCALE_FACTOR_SPATIAL,
                    width=width * VAE_SCALE_FACTOR_SPATIAL,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                    patch_size=model_config.patch_size,
                    patch_size_t=model_config.patch_size_t
                    if hasattr(model_config, "patch_size_t") else None,
                    attention_head_dim=model_config.attention_head_dim,
                    device=accelerator.device,
                    base_height=RoPE_BASE_HEIGHT,
                    base_width=RoPE_BASE_WIDTH,
                ) if model_config.use_rotary_positional_embeddings else None)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                # noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents, cond_latents],
                                              dim=2)
                model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config,
                                                                      "ofs_embed_dim") else None,
                ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full(
                    (1, ), fill_value=2.0)
                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                loss = torch.mean(
                    (weights * (model_pred - target)**2).reshape(batch_size, -1),
                    dim=1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())
                    logs.update({
                        "gradient_norm_before_clip": gradient_norm_before_clip,
                        "gradient_norm_after_clip": gradient_norm_after_clip,
                    })
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs.update({
                "loss": loss.detach().item(),
                "lr": args.learning_rate,
            })
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    name = Path(os.path.basename(__file__)).stem
    path_log = f'logs/{name}'
    os.makedirs(path_log, exist_ok=True)
    args = OmegaConf.load(f'configs/{name}.yaml')
    args.output_dir = path_log
    from socket import gethostname
    if gethostname() == "comar-System-Product-Name":
        args.data_root = args.data_root.replace('datasets/', 'datasets_local/')

    main(args)
