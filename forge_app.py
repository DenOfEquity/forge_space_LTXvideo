from diffusers.utils import check_min_version
check_min_version("0.32.0")

import spaces
import gradio as gr

import torch

from huggingface_hub import snapshot_download
from transformers import CLIPProcessor, CLIPModel

from transformers import T5EncoderModel, T5Tokenizer, BitsAndBytesConfig
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video

from pipeline_ltx_image2video import LTXImageToVideoPipeline

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors"


import numpy as np
import cv2
from PIL import Image
import os
import gc
import random

from datetime import datetime

MAX_SEED = np.iinfo(np.int32).max

# Global variables to load components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LTXStorage:
    noUnload = False
    lowVRAM = False
    pipelineTE = None
    pipelineTR = None
    lastPrompt = None
    lastNegative = None
    positive_embeds = None
    negative_embeds = None
    positive_attention = None
    negative_attention = None


def generate_video(
    image,
    prompt="",
    negative_prompt="",
    frame_rate=25,
    seed=11111,
    randomize_seed=False,
    num_inference_steps=30,
    guidance_scale=3,
    width=800,
    height=480,
    num_frames=41,
    progress=gr.Progress(),
):
    torch.set_grad_enabled(False)

    if len(prompt.strip()) < 50:
        raise gr.Error(
            "Prompt must be at least 50 characters long. Please provide more details for the best results.",
            duration=5,
        )

    ##  text encoding, if prompt has changed
    if prompt != LTXStorage.lastPrompt or negative_prompt != LTXStorage.lastNegative:
        if LTXStorage.pipelineTE is None:
            LTXStorage.pipelineTE = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                transformer=None,
                vae=None,
                scheduler=None,
                text_encoder=T5EncoderModel.from_pretrained("Lightricks/T5-XXL-8bit", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True),
                torch_dtype=torch.bfloat16,
            )

        if LTXStorage.lowVRAM == True:
            LTXStorage.pipelineTE.enable_sequential_cpu_offload()
        else:
            LTXStorage.pipelineTE.enable_model_cpu_offload()

        LTXStorage.prompt_embeds, LTXStorage.prompt_attention_mask, LTXStorage.negative_prompt_embeds, LTXStorage.negative_prompt_attention_mask = LTXStorage.pipelineTE.encode_prompt(prompt, negative_prompt=negative_prompt)

        LTXStorage.lastPrompt = prompt
        LTXStorage.lastNegative = negative_prompt

        if LTXStorage.noUnload == False:
            LTXStorage.pipelineTE = None
            torch.cuda.empty_cache()
            gc.collect()

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def gradio_progress_callback(self, step, timestep, kwargs):
        progress((step + 1) / num_inference_steps)

    try:
        if LTXStorage.pipelineTR is None:
            LTXStorage.pipelineTR = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                transformer=LTXVideoTransformer3DModel.from_single_file(single_file_url, torch_dtype=torch.bfloat16),
                vae=AutoencoderKLLTXVideo.from_single_file(single_file_url, torch_dtype=torch.bfloat16),
                text_encoder=None,
                tokenizer=None,
                torch_dtype=torch.bfloat16,
            )
            LTXStorage.pipelineTR.vae.enable_slicing()

        if LTXStorage.lowVRAM == True:
            LTXStorage.pipelineTR.enable_sequential_cpu_offload()
        else:
            LTXStorage.pipelineTR.to('cuda')

        video = LTXStorage.pipelineTR(
            prompt_embeds=LTXStorage.prompt_embeds.to('cuda'),
            prompt_attention_mask=LTXStorage.prompt_attention_mask.to('cuda'),
            negative_prompt_embeds=LTXStorage.negative_prompt_embeds.to('cuda'),
            negative_prompt_attention_mask=LTXStorage.negative_prompt_attention_mask.to('cuda'),
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            frame_rate=frame_rate,
            image=image,
            noUnload=LTXStorage.noUnload,
            lowVRAM=LTXStorage.lowVRAM,
        ).frames[0]

        if LTXStorage.noUnload == False:
            LTXStorage.pipelineTR = None

        output_path = f"outputs\\LTXvideo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        export_to_video(video, output_path+".mp4", fps=frame_rate)
        del video
        
        info = f"Prompt: {prompt}\nNegative prompt: {negative_prompt}\nSize: {width}x{height}, Frames (rate): {num_frames} ({frame_rate}), Steps: {num_inference_steps}, Seed: {seed}, Guidance: {guidance_scale}"
        with open(output_path+".txt", "w", encoding="utf8") as file:
            file.write(f"{info}\n")
    except Exception as e:
        raise gr.Error(
            f"An error occurred while generating the video. Please try again. Error: {e}",
            duration=5,
        )
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    return output_path+".mp4", seed


def unload():
    LTXStorage.pipelineTE = None
    LTXStorage.pipelineTR = None
    LTXStorage.lastPrompt = None
    LTXStorage.lastNegative = None
    LTXStorage.positive_embeds = None
    LTXStorage.negative_embeds = None
    LTXStorage.positive_attention = None
    LTXStorage.negative_attention = None
    gc.collect()
    torch.cuda.empty_cache()

def toggleNU ():
    LTXStorage.noUnload ^= True
    return gr.Button.update(variant=['secondary', 'primary'][LTXStorage.noUnload])
def toggleLV ():
    LTXStorage.lowVRAM = True   #   note: once applied, stays applied for this session
    return gr.Button.update(variant=['secondary', 'primary'][LTXStorage.lowVRAM])
def setWH (image):
    width = image.size[0]
    height = image.size[1]
    
    long = max(width, height)
    if long > 1280:
        w = width / long * 1280
        h = height / long * 1280
        width = 32 * (int(w + 16) // 32)
        height = 32 * (int(h + 16) // 32)
    
    return [width, height]

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
                height="40vh",
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

            with gr.Row():
                width_slider = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1280,
                    step=32,
                    value=800,
                    visible=True,
                )
                height_slider = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1280,
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
                    maximum=257,
                    step=8,
                    value=41,
                    visible=True,
                )

            with gr.Row():
                seed = gr.Number(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=11111)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=30)
                guidance_scale = gr.Slider(label="Guidance", minimum=1.0, maximum=5.0, step=0.1, value=3.0)

            with gr.Row():
                noUnload = gr.Button(value='keep models loaded', variant='primary' if LTXStorage.noUnload else 'secondary', tooltip='noUnload', scale=0)
                lowVRAM = gr.Button(value='low VRAM', variant='primary' if LTXStorage.lowVRAM else 'secondary', tooltip='low VRAM', scale=0)

        noUnload.click(toggleNU, inputs=[], outputs=noUnload)
        lowVRAM.click(toggleLV, inputs=[], outputs=lowVRAM)
        img2vid_image.upload(setWH, inputs=img2vid_image, outputs=[width_slider, height_slider])

        with gr.Column():
            txt2vid_generate = gr.Button(
                "Generate Video",
                variant="primary",
                size="lg",
            )
            txt2vid_output = gr.Video(label="Generated Output", height="70vh")

    txt2vid_generate.click(
        fn=generate_video,
        inputs=[
            img2vid_image,
            txt2vid_prompt,
            txt2vid_negative_prompt,
            txt2vid_frame_rate,
            seed,
            randomize_seed,
            inference_steps,
            guidance_scale,
            width_slider,
            height_slider,
            num_frames_slider            
        ],
        outputs=[txt2vid_output, seed],
        concurrency_limit=1,
        concurrency_id="generate_video",
        queue=True,
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

        - Seed: Save seed values to recreate specific styles or compositions you like
        - Guidance Scale: 3-3.5 are the recommended values
        - Inference Steps: More steps (40+) for quality, fewer steps (20-30) for speed
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
                    <img src="https://img.shields.io/badge/Project-Page-green">
                </a>
                <a href="https://huggingface.co/Lightricks">
                    Lightricks HuggingFace
                </a>
            </div>
            """
            )

    demo.unload(fn=unload)


if __name__ == "__main__":
    demo.queue(max_size=64, default_concurrency_limit=1, api_open=False).launch(show_api=False)

