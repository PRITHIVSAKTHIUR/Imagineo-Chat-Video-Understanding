import os
import random
import uuid
import json
import time
import asyncio
import re
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import edge_tts
import cv2

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from transformers.image_utils import load_image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# Constants
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
MAX_SEED = np.iinfo(np.int32).max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper function to return a progress bar HTML snippet.
def progress_bar_html(label: str) -> str:
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #F0FFF0; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #00FF00 ; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

# TEXT & TTS MODELS

model_id = "prithivMLmods/FastThink-0.5B-Tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

TTS_VOICES = [
    "en-US-JennyNeural",  # @tts1
    "en-US-GuyNeural",    # @tts2
]

# MULTIMODAL (OCR) MODELS

MODEL_ID_VL = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID_VL, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_VL,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

async def text_to_speech(text: str, voice: str, output_file="output.mp3"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def clean_chat_history(chat_history):
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

bad_words = json.loads(os.getenv('BAD_WORDS', "[]"))
bad_words_negative = json.loads(os.getenv('BAD_WORDS_NEGATIVE', "[]"))
default_negative = os.getenv("default_negative", "")

def check_text(prompt, negative=""):
    for i in bad_words:
        if i in prompt:
            return True
    for i in bad_words_negative:
        if i in negative:
            return True
    return False

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

dtype = torch.float16 if device.type == "cuda" else torch.float32

# STABLE DIFFUSION IMAGE GENERATION MODELS

if torch.cuda.is_available():
    # Lightning 5 model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False
    ).to(device)
    pipe.text_encoder = pipe.text_encoder.half()
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        print("Loaded RealVisXL_V5.0_Lightning on Device!")
    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        print("Model RealVisXL_V5.0_Lightning Compiled!")
    
    # Lightning 4 model
    pipe2 = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)
    pipe2.text_encoder = pipe2.text_encoder.half()
    if ENABLE_CPU_OFFLOAD:
        pipe2.enable_model_cpu_offload()
    else:
        pipe2.to(device)
        print("Loaded RealVisXL_V4.0 on Device!")
    if USE_TORCH_COMPILE:
        pipe2.unet = torch.compile(pipe2.unet, mode="reduce-overhead", fullgraph=True)
        print("Model RealVisXL_V4.0 Compiled!")
    
    # Turbo v3 model
    pipe3 = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0_Turbo",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)
    pipe3.text_encoder = pipe3.text_encoder.half()
    if ENABLE_CPU_OFFLOAD:
        pipe3.enable_model_cpu_offload()
    else:
        pipe3.to(device)
        print("Loaded RealVisXL_V3.0_Turbo on Device!")
    if USE_TORCH_COMPILE:
        pipe3.unet = torch.compile(pipe3.unet, mode="reduce-overhead", fullgraph=True)
        print("Model RealVisXL_V3.0_Turbo Compiled!")
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False
    ).to(device)
    pipe2 = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)
    pipe3 = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0_Turbo",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)
    print("Running on CPU; models loaded in float32.")

DEFAULT_MODEL = "Lightning 5"
MODEL_CHOICES = [DEFAULT_MODEL, "Lightning 4", "Turbo v3"]
models = {
    "Lightning 5": pipe,
    "Lightning 4": pipe2,
    "Turbo v3": pipe3
}

def save_image(img: Image.Image) -> str:
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

# GEMMA3-4B MULTIMODAL MODEL

gemma3_model_id = "google/gemma-3-4b-it"
gemma3_model = Gemma3ForConditionalGeneration.from_pretrained(
    gemma3_model_id, device_map="auto"
).eval()
gemma3_processor = AutoProcessor.from_pretrained(gemma3_model_id)

# VIDEO PROCESSING HELPER
def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    # Sample 10 evenly spaced frames.
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            # Convert from BGR to RGB and then to PIL Image.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

# MAIN GENERATION FUNCTION

@spaces.GPU
def generate(
    input_dict: dict,
    chat_history: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    text = input_dict["text"]
    files = input_dict.get("files", [])

    lower_text = text.lower().strip()

    # IMAGE GENERATION BRANCH (Stable Diffusion models)
    if (lower_text.startswith("@lightningv5") or 
        lower_text.startswith("@lightningv4") or 
        lower_text.startswith("@turbov3")):
        
        # Determine model choice based on flag.
        model_choice = None
        if "@lightningv5" in lower_text:
            model_choice = "Lightning 5"
        elif "@lightningv4" in lower_text:
            model_choice = "Lightning 4"
        elif "@turbov3" in lower_text:
            model_choice = "Turbo v3"
        
        # Remove the model flag from the prompt.
        prompt_clean = re.sub(r"@lightningv5", "", text, flags=re.IGNORECASE)
        prompt_clean = re.sub(r"@lightningv4", "", prompt_clean, flags=re.IGNORECASE)
        prompt_clean = re.sub(r"@turbov3", "", prompt_clean, flags=re.IGNORECASE)
        prompt_clean = prompt_clean.strip().strip('"')
        
        # Default parameters for single image generation.
        width = 1024
        height = 1024
        guidance_scale = 6.0
        seed_val = 0
        randomize_seed_flag = True
        
        seed_val = int(randomize_seed_fn(seed_val, randomize_seed_flag))
        generator = torch.Generator(device=device).manual_seed(seed_val)
        
        options = {
            "prompt": prompt_clean,
            "negative_prompt": default_negative,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": 30,
            "generator": generator,
            "num_images_per_prompt": 1,
            "use_resolution_binning": True,
            "output_type": "pil",
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        selected_pipe = models.get(model_choice, pipe)
        yield progress_bar_html("Processing Image Generation")
        images = selected_pipe(**options).images
        image_path = save_image(images[0])
        yield gr.Image(image_path)
        return

    # GEMMA3-4B TEXT & MULTIMODAL (image) Branch
    if lower_text.startswith("@gemma3-4b"):
        # If it is video, let the dedicated branch handle it.
        if lower_text.startswith("@gemma3-4b-video"):
            pass  # video branch is handled below.
        else:
            # Remove the gemma3 flag from the prompt.
            prompt_clean = re.sub(r"@gemma3-4b", "", text, flags=re.IGNORECASE).strip().strip('"')
            if files:
                # If image files are provided, load them.
                images = [load_image(f) for f in files]
                messages = [{
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": prompt_clean},
                    ]
                }]
            else:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
                ]
            inputs = gemma3_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(gemma3_model.device, dtype=torch.bfloat16)
            streamer = TextIteratorStreamer(
                gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            yield progress_bar_html("Processing with Gemma3-4b")
            for new_text in streamer:
                buffer += new_text
                time.sleep(0.01)
                yield buffer
            return

    # NEW: GEMMA3-4B VIDEO Branch
    if lower_text.startswith("@gemma3-4b-video"):
        # Remove the video flag from the prompt.
        prompt_clean = re.sub(r"@gemma3-4b-video", "", text, flags=re.IGNORECASE).strip().strip('"')
        if files:
            # Assume the first file is a video.
            video_path = files[0]
            frames = downsample_video(video_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]
            # Append each frame as an image with a timestamp label.
            for frame in frames:
                image, timestamp = frame
                # Save the frame image to a temporary unique filename.
                image_path = f"video_frame_{uuid.uuid4().hex}.png"
                image.save(image_path)
                messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
                messages[1]["content"].append({"type": "image", "url": image_path})
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]
        inputs = gemma3_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(gemma3_model.device, dtype=torch.bfloat16)
        streamer = TextIteratorStreamer(
            gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        yield progress_bar_html("Processing with Gemma3-4b Video")
        for new_text in streamer:
            buffer += new_text
            time.sleep(0.01)
            yield buffer
        return

    # Otherwise, handle text/chat (and TTS) generation.
    tts_prefix = "@tts"
    is_tts = any(text.strip().lower().startswith(f"{tts_prefix}{i}") for i in range(1, 3))
    voice_index = next((i for i in range(1, 3) if text.strip().lower().startswith(f"{tts_prefix}{i}")), None)
    
    if is_tts and voice_index:
        voice = TTS_VOICES[voice_index - 1]
        text = text.replace(f"{tts_prefix}{voice_index}", "").strip()
        conversation = [{"role": "user", "content": text}]
    else:
        voice = None
        text = text.replace(tts_prefix, "").strip()
        conversation = clean_chat_history(chat_history)
        conversation.append({"role": "user", "content": text})
    
    if files:
        images = [load_image(image) for image in files] if len(files) > 1 else [load_image(files[0])]
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text},
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        yield progress_bar_html("Processing with Qwen2VL Ocr")
        for new_text in streamer:
            buffer += new_text
            buffer = buffer.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
    else:
        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        }
        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        outputs = []
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)

        final_response = "".join(outputs)
        yield final_response

        if is_tts and voice:
            output_file = asyncio.run(text_to_speech(final_response, voice))
            yield gr.Audio(output_file, autoplay=True)

demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    examples=[
        [{"text": "@gemma3-4b Explain the Image", "files": ["examples/3.jpg"]}],
        [{"text": "@gemma3-4b-video Explain what is happening in this video ?", "files": ["examples/oreo.mp4"]}],
        [{"text": "@gemma3-4b-video Summarize the events in this video", "files": ["examples/sky.mp4"]}],
        [{"text": "@gemma3-4b-video What is in the video ?", "files": ["examples/redlight.mp4"]}],
        [{"text": "@gemma3-4b Where do the major drought happen?", "files": ["examples/111.png"]}],
        [{"text": "@gemma3-4b Transcription of the letter", "files": ["examples/222.png"]}],
        ['@lightningv5 Chocolate dripping from a donut'],
        ["Python Program for Array Rotation"],
        ["@tts1 Who is Nikola Tesla, and why did he die?"],
        ['@lightningv4 Cat holding a sign that says hello world'],
        ['@turbov3 Anime illustration of a wiener schnitzel'],
        ["@tts2 What causes rainbows to form?"],
    ],
    cache_examples=False,
    type="messages",
    description="# **Imagineo Chat `@gemma3-4b 'prompt..', @gemma3-4b-video, @lightningv5, etc..`**",
    fill_height=True,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image", "video"], file_count="multiple", placeholder="use the tags @gemma3-4b for multimodal, @gemma3-4b-video for video, @lightningv5, @lightningv4, @turbov3 for image gen !"),
    stop_btn="Stop Generation",
    multimodal=True,
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)