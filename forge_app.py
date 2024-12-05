import gradio as gr
#from gradio_toggle import Toggle
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPProcessor, CLIPModel
#from optimum.quanto import quantize, freeze, qint8

from xora.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from xora.models.transformers.transformer3d import Transformer3DModel
from xora.models.transformers.symmetric_patchifier import SymmetricPatchifier
from xora.schedulers.rf import RectifiedFlowScheduler
from xora.pipelines.pipeline_xora_video import XoraVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer
from xora.utils.conditioning_method import ConditioningMethod
from pathlib import Path
import safetensors.torch
import json
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import gc
import csv
from datetime import datetime

filenumber = 0

# Load Hugging Face token if needed
hf_token = os.getenv("HF_TOKEN")
system_prompt_t2v_path = "assets/system_prompt_t2v.txt"
system_prompt_i2v_path = "assets/system_prompt_i2v.txt"
with open(system_prompt_t2v_path, "r") as f:
    system_prompt_t2v = f.read()

with open(system_prompt_i2v_path, "r") as f:
    system_prompt_i2v = f.read()

# Set model download directory within Hugging Face Spaces
model_path = snapshot_download("Lightricks/LTX-Video", repo_type="model", ignore_patterns=['ltx-video-2b-v0.9.safetensors', 'media/*.*'])#, token=hf_token)

# Global variables to load components
vae_dir = Path(model_path) / "vae"
unet_dir = Path(model_path) / "unet"
scheduler_dir = Path(model_path) / "scheduler"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.float8_e4m3fn
# dtype = torch.bfloat16
dtype = torch.float16
dtype_vae = torch.float16

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def compute_clip_embedding(text=None, image=None):
    # """
    # Compute CLIP embedding for a given text or image.
    # Args:
        # text (str): Input text prompt.
        # image (PIL.Image): Input image.
    # Returns:
        # list: CLIP embedding as a list of floats.
    # """
    # inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)

    # outputs = clip_model.get_text_features(**inputs) if text else clip_model.get_image_features(**inputs)

    # embedding = outputs.detach().cpu().numpy().flatten().tolist()
    # return embedding


def load_vae(vae_dir):
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    return vae.to(dtype=dtype_vae)


def load_unet(unet_dir):
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    
    # quantize(transformer, weights=qint8, activations=qint8)
    # freeze(transformer)
    
    return transformer.to(dtype=dtype)


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


# Helper function for image processing
def center_crop_and_resize(frame, target_height, target_width):
    h, w, _ = frame.shape
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = w / h
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(h * aspect_ratio_target)
        x_start = (w - new_width) // 2
        frame_cropped = frame[:, x_start : x_start + new_width]
    else:
        new_height = int(w / aspect_ratio_target)
        y_start = (h - new_height) // 2
        frame_cropped = frame[y_start : y_start + new_height, :]
    frame_resized = cv2.resize(frame_cropped, (target_width, target_height))
    return frame_resized


def load_image_to_tensor_with_resize(image, target_height=512, target_width=768):
    image_np = np.array(image.convert("RGB"))
    frame_resized = center_crop_and_resize(image_np, target_height, target_width)
    frame_tensor = torch.tensor(frame_resized).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    return frame_tensor.unsqueeze(0).unsqueeze(2)


# Preset options for resolution and frame configuration
preset_options = [
    {"label": "1216x704, 41 frames", "width": 1216, "height": 704, "num_frames": 41},
    {"label": "1088x704, 49 frames", "width": 1088, "height": 704, "num_frames": 49},
    {"label": "1056x640, 57 frames", "width": 1056, "height": 640, "num_frames": 57},
    {"label": "992x608, 65 frames", "width": 992, "height": 608, "num_frames": 65},
    {"label": "896x608, 73 frames", "width": 896, "height": 608, "num_frames": 73},
    {"label": "896x544, 81 frames", "width": 896, "height": 544, "num_frames": 81},
    {"label": "832x544, 89 frames", "width": 832, "height": 544, "num_frames": 89},
    {"label": "800x512, 97 frames", "width": 800, "height": 512, "num_frames": 97},
    {"label": "768x512, 97 frames", "width": 768, "height": 512, "num_frames": 97},
    {"label": "800x480, 41 frames", "width": 800, "height": 480, "num_frames": 41},
    {"label": "800x480, 105 frames", "width": 800, "height": 480, "num_frames": 105},
    {"label": "736x480, 113 frames", "width": 736, "height": 480, "num_frames": 113},
    {"label": "704x480, 121 frames", "width": 704, "height": 480, "num_frames": 121},
    {"label": "704x448, 129 frames", "width": 704, "height": 448, "num_frames": 129},
    {"label": "672x448, 137 frames", "width": 672, "height": 448, "num_frames": 137},
    {"label": "640x416, 153 frames", "width": 640, "height": 416, "num_frames": 153},
    {"label": "672x384, 161 frames", "width": 672, "height": 384, "num_frames": 161},
    {"label": "640x384, 169 frames", "width": 640, "height": 384, "num_frames": 169},
    {"label": "608x384, 177 frames", "width": 608, "height": 384, "num_frames": 177},
    {"label": "576x384, 185 frames", "width": 576, "height": 384, "num_frames": 185},
    {"label": "608x352, 193 frames", "width": 608, "height": 352, "num_frames": 193},
    {"label": "576x352, 201 frames", "width": 576, "height": 352, "num_frames": 201},
    {"label": "544x352, 209 frames", "width": 544, "height": 352, "num_frames": 209},
    {"label": "512x352, 225 frames", "width": 512, "height": 352, "num_frames": 225},
    {"label": "512x352, 233 frames", "width": 512, "height": 352, "num_frames": 233},
    {"label": "544x320, 241 frames", "width": 544, "height": 320, "num_frames": 241},
    {"label": "512x320, 249 frames", "width": 512, "height": 320, "num_frames": 249},
    {"label": "512x320, 257 frames", "width": 512, "height": 320, "num_frames": 257},
]


# Function to toggle visibility of sliders based on preset selection
def preset_changed(preset):
    selected = next(item for item in preset_options if item["label"] == preset)
    return (
        selected["height"],
        selected["width"],
        selected["num_frames"],
    )

# Load models
vae = load_vae(vae_dir)
unet = load_unet(unet_dir)
scheduler = load_scheduler(scheduler_dir)
patchifier = SymmetricPatchifier(patch_size=1)
try:
    text_encoder = T5EncoderModel.from_pretrained("./pixart_T5_fp16", variant="fp16", local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
except:
    text_encoder = T5EncoderModel.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="text_encoder").to(torch.float16)
tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="tokenizer")

pipeline = XoraVideoPipeline(
    transformer=unet,
    patchifier=patchifier,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    vae=vae,
)#.to(device)
#pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()
#vae slicing/tiling not functional


def generate_video_from_text(
    prompt="",
    negative_prompt="",
    frame_rate=25,
    seed=646373,
    num_inference_steps=30,
    guidance_scale=3,
    height=480,
    width=800,
    num_frames=41,
    progress=gr.Progress(),
):
    if len(prompt.strip()) < 50:
        raise gr.Error(
            "Prompt must be at least 50 characters long. Please provide more details for the best results.",
            duration=5,
        )

    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": None,
    }

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def gradio_progress_callback(self, step, timestep, kwargs):
        progress((step + 1) / num_inference_steps)

    try:
        with torch.no_grad():
            images = pipeline(
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                **sample,
                is_video=True,
                vae_per_channel_normalize=True,
                conditioning_method=ConditioningMethod.UNCONDITIONAL,
                mixed_precision=True,
                callback_on_step_end=gradio_progress_callback,
            ).images
    except Exception as e:
        raise gr.Error(
            f"An error occurred while generating the video. Please try again. Error: {e}",
            duration=5,
        )
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    global filenumber
    output_path = f"outputs\\LTXvideo_{filenumber}.mp4"
    filenumber += 1
#    print(images.shape)
    video_np = images.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    height, width = video_np.shape[1:3]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))
    for frame in video_np[..., ::-1]:
        out.write(frame)
    out.release()
    # Explicitly delete tensors and clear cache
    del images
    del video_np
    torch.cuda.empty_cache()
    return output_path


def generate_video_from_image(
    image,
    prompt="",
    negative_prompt="",
    frame_rate=25,
    seed=646373,
    num_inference_steps=30,
    guidance_scale=3,
    height=480,
    width=800,
    num_frames=41,
    progress=gr.Progress(),
):

    if len(prompt.strip()) < 50:
        raise gr.Error(
            "Prompt must be at least 50 characters long. Please provide more details for the best results.",
            duration=5,
        )

    if not image:
        raise gr.Error("Please provide an input image.", duration=5)

    media_items = load_image_to_tensor_with_resize(image, height, width).to(device).detach()

    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def gradio_progress_callback(self, step, timestep, kwargs):
        progress((step + 1) / num_inference_steps)

    try:
        with torch.no_grad():
            images = pipeline(
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                **sample,
                is_video=True,
                vae_per_channel_normalize=True,
                conditioning_method=ConditioningMethod.FIRST_FRAME,
                mixed_precision=True,
                callback_on_step_end=gradio_progress_callback,
            ).images

        global filenumber
        output_path = f"outputs\\LTXvideo_{filenumber}.mp4"
        filenumber += 1

        video_np = images.squeeze(0).permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        height, width = video_np.shape[1:3]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))
        for frame in video_np[..., ::-1]:
            out.write(frame)
        out.release()
    except Exception as e:
        raise gr.Error(
            f"An error occurred while generating the video. Please try again. Error: {e}",
            duration=5,
        )

    finally:
        torch.cuda.empty_cache()
        gc.collect()

    return output_path


def create_advanced_options():
    with gr.Accordion("Step 4: Advanced Options (Optional)", open=True):
        seed = gr.Slider(label="4.1 Seed", minimum=0, maximum=1000000, step=1, value=646373)
        inference_steps = gr.Slider(label="4.2 Inference Steps", minimum=1, maximum=50, step=1, value=30)
        guidance_scale = gr.Slider(label="4.3 Guidance Scale", minimum=1.0, maximum=5.0, step=0.1, value=3.0)

        height_slider = gr.Slider(
            label="4.4 Height",
            minimum=256,
            maximum=1024,
            step=32,
            value=480,
            visible=True,
        )
        width_slider = gr.Slider(
            label="4.5 Width",
            minimum=256,
            maximum=1024,
            step=32,
            value=800,
            visible=True,
        )
        num_frames_slider = gr.Slider(
            label="4.5 Number of Frames",
            minimum=1,
            maximum=200,
            step=1,
            value=41,
            visible=True,
        )

        return [
            seed,
            inference_steps,
            guidance_scale,
            height_slider,
            width_slider,
            num_frames_slider,
        ]


# Define the Gradio interface with tabs
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row(elem_id="title-row"):
        gr.Markdown(
            """
        <div style="text-align: center; margin-bottom: 1em">
            <h1 style="font-size: 2.5em; font-weight: 600; margin: 0.5em 0;">Video Generation with LTX Video</h1>
        </div>
        """
        )
    with gr.Row(elem_id="title-row"):
        gr.HTML(  # add technical report link
            """
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/Lightricks/LTX-Video">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a>
            <a href="http://www.lightricks.com/ltxv">
                <img src="https://img.shields.io/badge/Project-Page-green" alt="Follow me on HF">
            </a>
            <a href="https://huggingface.co/Lightricks">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
            </a>
        </div>
        """
        )
    with gr.Accordion(" 📖 Tips for Best Results", open=False, elem_id="instructions-accordion"):
        gr.Markdown(
            """
        📝 Prompt Engineering

        When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words.
        For best results, build your prompts using this structure:

        - Start with main action in a single sentence
        - Add specific details about movements and gestures
        - Describe character/object appearances precisely
        - Include background and environment details
        - Specify camera angles and movements
        - Describe lighting and colors
        - Note any changes or sudden events

        See examples for more inspiration.

        🎮 Parameter Guide

        - Resolution Preset: Higher resolutions for detailed scenes, lower for faster generation and simpler scenes
        - Seed: Save seed values to recreate specific styles or compositions you like
        - Guidance Scale: 3-3.5 are the recommended values
        - Inference Steps: More steps (40+) for quality, fewer steps (20-30) for speed
        """
        )

    with gr.Tabs():
        # Text to Video Tab
        with gr.TabItem("Text to Video"):
            with gr.Row():
                with gr.Column():
                    txt2vid_prompt = gr.Textbox(
                        label="Step 1: Enter Your Prompt",
                        placeholder="Describe the video you want to generate (minimum 50 characters)...",
                        value="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage.",
                        lines=5,
                    )

                    txt2vid_negative_prompt = gr.Textbox(
                        label="Step 2: Enter Negative Prompt",
                        placeholder="Describe what you don't want in the video...",
                        value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                        lines=2,
                    )

                    txt2vid_preset = gr.Dropdown(
                        choices=[p["label"] for p in preset_options],
                        value="800x480, 41 frames",
                        label="Step 3.1: Choose Resolution Preset",
                    )

                    txt2vid_frame_rate = gr.Slider(
                        label="Step 3.2: Frame Rate",
                        minimum=21,
                        maximum=30,
                        step=1,
                        value=25,
                    )

                    txt2vid_advanced = create_advanced_options()
                    txt2vid_generate = gr.Button(
                        "Step 5: Generate Video",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column():
                    txt2vid_output = gr.Video(label="Generated Output")

            with gr.Row():
                gr.Examples(
                    examples=[
                        [
                            "A young woman in a traditional Mongolian dress is peeking through a sheer white curtain, her face showing a mix of curiosity and apprehension. The woman has long black hair styled in two braids, adorned with white beads, and her eyes are wide with a hint of surprise. Her dress is a vibrant blue with intricate gold embroidery, and she wears a matching headband with a similar design. The background is a simple white curtain, which creates a sense of mystery and intrigue.ith long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair’s face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                            "assets/t2v_2.mp4",
                        ],
                        [
                            "A young man with blond hair wearing a yellow jacket stands in a forest and looks around. He has light skin and his hair is styled with a middle part. He looks to the left and then to the right, his gaze lingering in each direction. The camera angle is low, looking up at the man, and remains stationary throughout the video. The background is slightly out of focus, with green trees and the sun shining brightly behind the man. The lighting is natural and warm, with the sun creating a lens flare that moves across the man’s face. The scene is captured in real-life footage.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                            "assets/t2v_1.mp4",
                        ],
                        [
                            "A cyclist races along a winding mountain road. Clad in aerodynamic gear, he pedals intensely, sweat glistening on his brow. The camera alternates between close-ups of his determined expression and wide shots of the breathtaking landscape. Pine trees blur past, and the sky is a crisp blue. The scene is invigorating and competitive.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                            "assets/t2v_0.mp4",
                        ],
                    ],
                    inputs=[txt2vid_prompt, txt2vid_negative_prompt, txt2vid_output],
                    label="Example Text-to-Video Generations",
                )

        # Image to Video Tab
        with gr.TabItem("Image to Video"):
            with gr.Row():
                with gr.Column():
                    img2vid_image = gr.Image(
                        type="pil",
                        label="Step 1: Upload Input Image",
                        elem_id="image_upload",
                    )
                    img2vid_prompt = gr.Textbox(
                        label="Step 2: Enter Your Prompt",
                        placeholder="Describe how you want to animate the image (minimum 50 characters)...",
                        value="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage.",
                        lines=5,
                    )
                    img2vid_negative_prompt = gr.Textbox(
                        label="Step 3: Enter Negative Prompt",
                        placeholder="Describe what you don't want in the video...",
                        value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                        lines=2,
                    )

                    img2vid_preset = gr.Dropdown(
                        choices=[p["label"] for p in preset_options],
                        value="800x480, 41 frames",
                        label="Step 3.1: Choose Resolution Preset",
                    )

                    img2vid_frame_rate = gr.Slider(
                        label="Step 3.2: Frame Rate",
                        minimum=21,
                        maximum=30,
                        step=1,
                        value=25,
                    )

                    img2vid_advanced = create_advanced_options()
                    img2vid_generate = gr.Button("Step 6: Generate Video", variant="primary", size="lg")

                with gr.Column():
                    img2vid_output = gr.Video(label="Generated Output")

            with gr.Row():
                gr.Examples(
                    examples=[
                        [
                            "assets/i2v_i2.png",
                            "A woman stirs a pot of boiling water on a white electric burner. Her hands, with purple nail polish, hold a wooden spoon and move it in a circular motion within a white pot filled with bubbling water. The pot sits on a white electric burner with black buttons and a digital display. The burner is positioned on a white countertop with a red and white checkered cloth partially visible in the bottom right corner. The camera angle is a direct overhead shot, remaining stationary throughout the scene. The lighting is bright and even, illuminating the scene with a neutral white light. The scene is real-life footage.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                            "assets/i2v_2.mp4",
                        ],
                        [
                            "assets/i2v_i0.png",
                            "A woman in a long, flowing dress stands in a field, her back to the camera, gazing towards the horizon; her hair is long and light, cascading down her back; she stands beneath the sprawling branches of a large oak tree;  to her left, a classic American car is parked on the dry grass; in the distance, a wrecked car lies on its side; the sky above is a dramatic canvas of bright white clouds against a darker sky; the entire image is in black and white, emphasizing the contrast of light and shadow. The woman is walking slowly towards the car.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                            "assets/i2v_0.mp4",
                        ],
                        [
                            "assets/i2v_i1.png",
                            "A pair of hands shapes a piece of clay on a pottery wheel, gradually forming a cone shape. The hands, belonging to a person out of frame, are covered in clay and gently press a ball of clay onto the center of a spinning pottery wheel. The hands move in a circular motion, gradually forming a cone shape at the top of the clay. The camera is positioned directly above the pottery wheel, providing a bird’s-eye view of the clay being shaped. The lighting is bright and even, illuminating the clay and the hands working on it. The scene is captured in real-life footage.",
                            "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                            "assets/i2v_1.mp4",
                        ],
                    ],
                    inputs=[
                        img2vid_image,
                        img2vid_prompt,
                        img2vid_negative_prompt,
                        img2vid_output,
                    ],
                    label="Example Image-to-Video Generations",
                )

    # [Previous event handlers remain the same]
    txt2vid_preset.change(fn=preset_changed, inputs=[txt2vid_preset], outputs=txt2vid_advanced[3:])

    txt2vid_generate.click(
        fn=generate_video_from_text,
        inputs=[
            txt2vid_prompt,
            txt2vid_negative_prompt,
            txt2vid_frame_rate,
            *txt2vid_advanced,
        ],
        outputs=txt2vid_output,
        concurrency_limit=1,
        concurrency_id="generate_video",
        queue=True,
    )

    img2vid_preset.change(fn=preset_changed, inputs=[img2vid_preset], outputs=img2vid_advanced[3:])

    img2vid_generate.click(
        fn=generate_video_from_image,
        inputs=[
            img2vid_image,
            img2vid_prompt,
            img2vid_negative_prompt,
            img2vid_frame_rate,
            *img2vid_advanced,
        ],
        outputs=img2vid_output,
        concurrency_limit=1,
        concurrency_id="generate_video",
        queue=True,
    )

if __name__ == "__main__":
    demo.queue(max_size=64, default_concurrency_limit=1, api_open=False).launch(show_api=False)

