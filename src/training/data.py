import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from decord import VideoReader, cpu
import re
import numpy as np

from .params import DataArguments
from .constants import *

def encode_video(video_path, max_num_frames=10):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(images, processor):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    content = []

    for img in images:
        content.append({"type": "image", "image": img})
    
    messages = [
        {
            "role": "user", 
            "content": content
        }
    ]

    vision_infos = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt")

    pixel_values = vision_infos["pixel_values"]

    return pixel_values

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False
        num_frames = None
        pixel_values = None

        processor = self.processor
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))
            
            pixel_values = get_image_info(images, processor)

        elif "video" in sources:
            video_file = sources["video"]
            video_folder = self.data_args.image_folder

            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)

            images = encode_video(video_file, self.max_num_frames)
            
            is_video = True
            num_frames = len(images)
            pixel_values = get_image_info(images, processor)

        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames))

        all_input_ids = [torch.tensor([2])] # bos token id
        all_labels = [torch.tensor([-100])] # ignore bos token

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            if idx == 0 and len(SYSTEM_MESSAGE) > 0:
                user_input = f"{DEFAULT_START_TOKEN}{user_input['role']}\n{SYSTEM_MESSAGE}\n\n{user_input['content']}{DEFAULT_END_TOKEN}\n{DEFAULT_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}{DEFAULT_END_TOKEN}\n"

            else:
                user_input = f"{DEFAULT_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_END_TOKEN}\n{DEFAULT_START_TOKEN}{gpt_response['role']}\n"
                gpt_response = f"{gpt_response['content']}{DEFAULT_END_TOKEN}\n"

            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos tokens in the input_ids
        # Gemma3 does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_values is not None:
            array_ids = input_ids
            token_type_ids = np.zeros_like(input_ids)
            token_type_ids[array_ids == processor.image_token_id] = 1
            token_type_ids = torch.tensor(token_type_ids)

            data_dict["pixel_values"] = pixel_values
            data_dict["token_type_ids"] = token_type_ids
            
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_token_type_ids = []
        
        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_token_type_ids.append(example["token_type_ids"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        
        batch_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            token_type_ids = pad_sequence(batch_token_type_ids, padding_side='right', padding_value=0)
            batch_dict.update(pixel_values=pixel_values, token_type_ids=token_type_ids)

        return batch_dict

def video_to_image_tokens(input_string, num_frames):
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)

    return input_string

def replace_image_tokens(input_string):
    pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
    replacement = "\n\n" + VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN*256 + VISION_END_TOKEN + "\n\n"

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False, num_frames=None):

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        
        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)
        
        transformed_content = replace_image_tokens(conversation["value"])
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)