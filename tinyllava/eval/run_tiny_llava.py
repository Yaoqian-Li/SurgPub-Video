import argparse
import re
import requests
from PIL import Image
from io import BytesIO
import csv
import os
import time
from tqdm import tqdm
import torch
from transformers import PreTrainedModel
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import functional as F
from torchvision.io import read_video

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
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


def normalize_choice(text):
    text = (text or "").strip().upper()
    if not text:
        return ""
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return text


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
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
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
    rows = []
    correct = 0
    total = 0
    missing = 0
    with open(args.input_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['outputs']+['frames']+['acc']
        
        for i, row in enumerate(tqdm(reader), 1):
            # row['row_number'] = i

            qs = row['question_closed']+f" A:{row['A']}, B:{row['B']}, C:{row['C']}, D:{row['D']}. "+'Please dirctly give the choice (A/B/C/D) without other content.'
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            frame_files = get_frames(
                row.get("path", ""),
                row["folder"],
                row["start"],
                row["end"],
                args.video_folder,
                args.frame_dir_name,
                args.frame_name_format,
                args.num_frame,
            )
            row['frames']=''


            msg = Message()
            msg.add_message(qs)

            result = text_processor(msg.messages, mode='eval')
            # 
            input_ids = result['input_ids']
            prompt = result['prompt']
            input_ids = input_ids.unsqueeze(0).cuda()
            
            images_tensor = None
            video_tensor = None
            if not os.path.exists(frame_files[-1]):
                row['outputs']='none'
                row['frames']=frame_files
                if 'answer_key' in row:
                    row['acc'] = False
                    total += 1
                rows.append(row)
                missing += 1
                continue
            frames = []

            for frame_file in frame_files:

                frame = Image.open(frame_file).convert('RGB')
                frame = image_preprocess(frame)
                frames.append(frame)

            video_tensor= torch.stack(frames).unsqueeze(dim=0)

                   


            stop_str = text_processor.template.separator.apply()[1]
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                print("tokenizer.pad_token_id:",tokenizer.pad_token_id)
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

            outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            row['outputs']=outputs
            row['frames']=frame_files
            if 'answer_key' in row:
                row['acc'] = normalize_choice(outputs) == normalize_choice(row['answer_key'])
                correct += int(row['acc'])
                total += 1
            print("result:",result)
            print("output:",outputs)
            rows.append(row)
            if i % args.save_step == 0:
                with open(args.output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
    
    if rows:
        with open(args.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")
    if missing:
        print(f"Missing-frame samples: {missing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--frame_dir_name", type=str, default="frames")
    parser.add_argument("--frame_name_format", type=str, default="frame_{:05d}.png")
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
    
    args = parser.parse_args()

    eval_model(args)
