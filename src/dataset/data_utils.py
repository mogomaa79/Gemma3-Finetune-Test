from PIL import Image
from decord import VideoReader, cpu
import re

from src.constants import (
    LLAVA_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

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