# Training GGI Module

## 1. Checkpoint

To train the **GGI module**, we use the pretrained model [**Go-with-the-Flow**](https://github.com/Eyeline-Labs/Go-with-the-Flow), which incorporates optical flow–aware guidance.

Please follow the steps below:

1. Go to [https://huggingface.co/Eyeline-Labs/Go-with-the-Flow/tree/main](https://huggingface.co/Eyeline-Labs/Go-with-the-Flow/tree/main)
2. Download the file: `I2V5B_final_i38800_nearest_lora_weights.safetensors`
3. Move the file to the following directory: `./training_ggi/lora_models/`

---

## 2. Dataset

We use the [**DL3DV-10K**](https://dl3dv-10k.github.io/DL3DV-10K/) dataset to train the module.
Please download the dataset and place it under:

```
datasets/DL3DV-10K
```

> **Note:**
>
> * Download the **1920×1080** resolution version (not full resolution).
> * Skip the COLMAP reconstruction data.

The dataset directory should also contain two supporting files:

* `datasets/seeds.txt`
* `datasets/captions.json`

### ⚙️ `datasets/seeds.txt`

DL3DV-10K contains some corrupted samples that cause severe training failures.
So, we store valid scene IDs in `seeds.txt`, which is automatically loaded by the training code.

### 🗒️ `datasets/captions.json`

To remove the need for prompt estimation during training, we pre-compute prompts from the first frame and store them in this file.
However, we found that training works well even without prompts. you may remove this file and slightly modify the training code to disable prompt loading.

Example directory structure:

```
DL3DV-10K
└── 1K
    ├── 001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f
    │   └── images_2
    └── 0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac
        └── images_2
```

## 3. Training

You can adjust hyperparameters such as `batch_size` and `gradient_accumulation` in the configuration file:

```
configs/T073-000.yaml
```

Once the checkpoint and dataset are prepared, start training using the following command:

```bash
cd training_ggi

# NUM_GPU : number of GPUs to use
accelerate launch --config_file 'default.yaml' \
                  --num_processes ${NUM_GPU} \
                  --main_process_port=4408 \
                  exps/T073-000.py
```

The trained module checkpoints will be saved automatically to the `./log` directory.

```
```

