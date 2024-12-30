# needs diffusers 0.32.0.dev or higher
from diffusers.utils import check_min_version
check_min_version("0.32.0")

import spaces
import gradio as gr
#from gradio_toggle import Toggle
import torch

from huggingface_hub import snapshot_download
from transformers import CLIPProcessor, CLIPModel

from transformers import T5EncoderModel, T5Tokenizer, BitsAndBytesConfig
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video

from pipeline_ltx_image2video import LTXImageToVideoPipeline

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors"

pipeline = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    transformer=LTXVideoTransformer3DModel.from_single_file(single_file_url, torch_dtype=torch.bfloat16),
    vae=AutoencoderKLLTXVideo.from_single_file(single_file_url, torch_dtype=torch.bfloat16),
    text_encoder=T5EncoderModel.from_pretrained("Lightricks/T5-XXL-8bit", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True),
    torch_dtype=torch.bfloat16
)

pipeline.vae.enable_slicing()
pipeline.enable_sequential_cpu_offload()

from enum import Enum
class ConditioningMethod(Enum):
    UNCONDITIONAL = "unconditional"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    FIRST_AND_LAST_FRAME = "first_and_last_frame"

import numpy as np
import cv2
from PIL import Image
import os
import gc

from datetime import datetime

# Global variables to load components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#patchifier = SymmetricPatchifier(patch_size=1)


def generate_video(
    image,
    prompt="",
    negative_prompt="",
    frame_rate=25,
    seed=646373,
    num_inference_steps=30,
    guidance_scale=3,
    width=800,
    height=480,
    num_frames=41,
    progress=gr.Progress(),
):

    if len(prompt.strip()) < 50:
        raise gr.Error(
            "Prompt must be at least 50 characters long. Please provide more details for the best results.",
            duration=5,
        )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def gradio_progress_callback(self, step, timestep, kwargs):
        progress((step + 1) / num_inference_steps)

    try:
        with torch.no_grad():
            video = pipeline(
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                frame_rate=frame_rate,
                image=image,
            ).frames[0]

        output_path = f"outputs\\LTXvideo_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        export_to_video(video, output_path, fps=frame_rate)
        del video

    except Exception as e:
        raise gr.Error(
            f"An error occurred while generating the video. Please try again. Error: {e}",
            duration=5,
        )

    finally:
        torch.cuda.empty_cache()
        gc.collect()

    return output_path


def unload():
    global pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


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

    with gr.Row():
        with gr.Column():
            img2vid_image = gr.Image(
                type="pil",
                height="50vh",
                label="Upload Input Image (leave empty for Text-to-Video)",
                elem_id="image_upload",
            )
            txt2vid_prompt = gr.Textbox(
                label="Enter Your Prompt",
                placeholder="Describe the video you want to generate (minimum 50 characters)...",
                value="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage.",
                lines=5,
            )

            txt2vid_negative_prompt = gr.Textbox(
                label="Enter Negative Prompt",
                placeholder="Describe what you don't want in the video...",
                value="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
                lines=2,
            )

            txt2vid_preset = gr.Dropdown(
                choices=[p["label"] for p in preset_options],
                value="800x480, 41 frames",
                label="Choose Resolution Preset",
            )

            with gr.Row():
                width_slider = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1024,
                    step=32,
                    value=800,
                    visible=True,
                )
                height_slider = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    step=32,
                    value=480,
                    visible=True,
                )

            with gr.Row():
                txt2vid_frame_rate = gr.Slider(
                    label="Frame Rate",
                    minimum=21,
                    maximum=30,
                    step=1,
                    value=25,
                )
                num_frames_slider = gr.Slider(
                    label="Number of Frames",
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=41,
                    visible=True,
                )

            with gr.Row():
                seed = gr.Number(label="Seed", minimum=0, maximum=1000000, step=1, value=646373, scale=1)
                inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=30)
                guidance_scale = gr.Slider(label="Guidance", minimum=1.0, maximum=5.0, step=0.1, value=3.0)

        with gr.Column():
            txt2vid_generate = gr.Button(
                "Generate Video",
                variant="primary",
                size="lg",
            )
            txt2vid_output = gr.Video(label="Generated Output")

    txt2vid_preset.change(fn=preset_changed, inputs=[txt2vid_preset], outputs=[width_slider, height_slider, num_frames_slider])

    txt2vid_generate.click(
        fn=generate_video,
        inputs=[
            img2vid_image,
            txt2vid_prompt,
            txt2vid_negative_prompt,
            txt2vid_frame_rate,
            seed,
            inference_steps,
            guidance_scale,
            width_slider,
            height_slider,
            num_frames_slider            
        ],
        outputs=txt2vid_output,
        concurrency_limit=1,
        concurrency_id="generate_video",
        queue=True,
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

    demo.unload(fn=unload)


if __name__ == "__main__":
    demo.queue(max_size=64, default_concurrency_limit=1, api_open=False).launch(show_api=False)

