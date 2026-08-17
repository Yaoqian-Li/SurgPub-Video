import argparse
import re
import requests
from PIL import Image
from io import BytesIO
import csv
import json
import os
import time
from tqdm import tqdm
import torch
from transformers import PreTrainedModel
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import functional as F
from torchvision.io import read_video
import pandas as pd
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
from tinyllava.model.load_model import apply_runtime_divprune_config
import numpy as np
import subprocess
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def video_parser(args):
    out = args.video_file.split(args.sep)
    return out

def get_n_elements_equally_spaced(lst, n):
    if not lst or n <= 0:
        return []
    
    length = len(lst)
    if length <= n:
        return lst[:]
    
    step = (length - 1) / (n - 1)
    result = []
    
    for i in range(n):
        index = round(i * step)
        result.append(lst[index])
    
    return result
def get_frames(path, folder, start, end, video_root, frame_dir_name, frame_name_format, num_frames):
    start_num = int(float(start))
    end_num = int(float(end))
    parts = [video_root]
    if path:
        parts.append(path)
    parts.extend([folder, frame_dir_name])
    frame_dir = os.path.join(*parts)
    frames = [os.path.join(frame_dir, frame_name_format.format(num)) for num in range(start_num, end_num + 1)]
    frames = get_n_elements_equally_spaced(frames, num_frames)
    return frames
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def configure_divprune(model, args):
    apply_runtime_divprune_config(
        model,
        enable_divprune=args.enable_divprune,
        divprune_keep_ratio=args.divprune_keep_ratio,
        divprune_keep_tokens=args.divprune_keep_tokens,
        divprune_preserve_order=args.divprune_preserve_order,
    )


def extract_divprune_stats(model):
    if hasattr(model, 'get_divprune_stats'):
        return model.get_divprune_stats()
    return None


def normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())


def normalize_action(text):
    text = str(text or "").strip()
    match = re.search(r"\b([A-H])\b", text.upper())
    if match:
        return match.group(1)
    return normalize_text(text)


def save_frames(frames, save_dir):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray((frame.cpu().numpy().transpose(1, 2, 0)).astype('uint8'))
        img.save(os.path.join(save_dir, f"frame_{i}.png"))


def eval_model(args):
    # Model
    disable_torch_init()

    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model(args.model_path)
        configure_divprune(model, args)
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        configure_divprune(model, args)
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
    tokenizer = model.tokenizer
    image_processor = model.vision_tower._image_processor
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_preprocess = ImagePreprocess(image_processor, data_args)
    video_preprocess = VideoPreprocess(image_processor, data_args)

    model.cuda()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.report_divprune_stats and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = []
    correct = 0
    total = 0
    missing = 0
    actions=['A. Other','B. Picking-up the needle','C. Positioning the needle tip','D. Pushingtheneedle through the tissue','E. Pullingthe needle out of the tissue','F. Tyingaknot','G. Cuttingthe suture','H. Returning/dropping the needle']

    with open(args.input_path, 'r') as f:
        data = json.load(f)

                   
        for i, row in enumerate(tqdm(data), 1):

            
 
            qs=DEFAULT_IMAGE_TOKEN + "\n" +'What Action related to the needle and suture is the surgeon focusing on right now? The available action options are A. Other B. Picking-up the needle C. Positioning the needle tip D. Pushingtheneedle through the tissue E. Pullingthe needle out of the tissue F. Tyingaknot G. Cuttingthe suture H. Returning/dropping the needle'

            frame_files = [os.path.join(args.video_folder, frame) for frame in row['frames']]
            if args.model_path and 'single' in args.model_path:
                frame_files=[frame_files[8]]
            msg = Message()
            msg.add_message(qs)

            result = text_processor(msg.messages, mode='eval')
            input_ids = result['input_ids']
            prompt = result['prompt']
            input_ids = input_ids.unsqueeze(0).cuda()
            
            images_tensor = None
            video_tensor = None
            if not os.path.exists(frame_files[-1]):
                row['outputs']='none'
                row['frames']=frame_files
                row['acc']=False
                rows.append(row)
                total += 1
                missing += 1
                continue
            frames = []
            for frame_file in frame_files:

                frame = Image.open(frame_file).convert('RGB')
                frame = image_preprocess(frame)
                frames.append(frame)

            video_tensor= torch.stack(frames).unsqueeze(dim=0)
            if args.report_divprune_stats and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            stop_str = text_processor.template.separator.apply()[1]
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            start_time = time.perf_counter()
            with torch.inference_mode():
                print("tokenizer.pad_token_id:",tokenizer.pad_token_id)
                generate_start = time.perf_counter()
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    video=video_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                generate_latency = time.perf_counter() - generate_start
            end_to_end_latency = time.perf_counter() - start_time

            outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            row['outputs']=outputs
            row['frames']=frame_files
            print("result:",result)
            print("output:",outputs)
            answer = actions[int(row['label'])]
            print('answer:',answer)
            row['acc'] = normalize_action(answer) == normalize_action(outputs)
            correct += int(row['acc'])
            total += 1
            if args.report_divprune_stats:
                stats = extract_divprune_stats(model) or {}
                row['divprune_enabled'] = stats.get('enabled', False)
                row['original_visual_tokens'] = stats.get('original_tokens')
                row['kept_visual_tokens'] = stats.get('kept_tokens')
                row['actual_keep_ratio'] = stats.get('actual_keep_ratio')
                row['end_to_end_latency_sec'] = end_to_end_latency
                row['generate_latency_sec'] = generate_latency
                if torch.cuda.is_available():
                    row['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            rows.append(row)
            if i % args.save_step == 0:
                output_df = pd.DataFrame(rows)
                output_df.to_csv(args.output_path, index=False)
    if rows:
        output_df = pd.DataFrame(rows)
        output_df.to_csv(args.output_path, index=False)
    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")
    if missing:
        print(f"Missing-frame samples: {missing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=PreTrainedModel, default=None)
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enable-divprune", dest="enable_divprune", action="store_true")
    parser.add_argument("--disable-divprune", dest="enable_divprune", action="store_false")
    parser.set_defaults(enable_divprune=None)
    parser.add_argument("--divprune-keep-ratio", type=float, default=None)
    parser.add_argument("--divprune-keep-tokens", type=int, default=None)
    parser.add_argument("--divprune-preserve-order", dest="divprune_preserve_order", action="store_true")
    parser.add_argument("--divprune-no-preserve-order", dest="divprune_preserve_order", action="store_false")
    parser.set_defaults(divprune_preserve_order=None)
    parser.add_argument("--report-divprune-stats", action="store_true")
    args = parser.parse_args()

    eval_model(args)
