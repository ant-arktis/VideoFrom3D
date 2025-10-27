from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from glob import glob
from tqdm import tqdm
from os.path import join
import json
from omegaconf import OmegaConf


def get_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                          torch_dtype=torch.float16)
    model.to(device)

    return model, processor


def main():

    device = 'cuda'
    model, processor = get_model()

    mat_cnt = 10000000000000000000

    caps = {}

    with open('seeds.txt', 'r') as f:
        targets = f.read().splitlines()

    for i, target in tqdm(enumerate(targets)):
        path = sorted(glob(f'{target}/images*/*.png'))[0]
        image = Image.open(path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        id = target.split('/')[-1]
        caps[id] = generated_text

        if mat_cnt < i:
            break

    with open('captions.json', 'w') as f:
        json.dump(caps, f)


if __name__ == "__main__":
    main()
