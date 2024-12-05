## Forge2 Spaces implementation of LTX-video ##
New Forge only.

derived from https://huggingface.co/spaces/Lightricks/LTX-Video-Playground

will download ~8.7GB models (+ ~18GB text encoder from PixArt, unless you already have it)

>[!NOTE]
>Install via *Extensions* tab; *Install from URL* sub-tab; use URL of this repo.
>Initial load can be *slow*.

>[!TIP]
>If you have sufficient VRAM, you may be able to improve performance by changing lines 190-191 of *forge_app.py* in the extension folder to:
```
pipeline.enable_model_cpu_offload()
#pipeline.enable_sequential_cpu_offload()
```

>[!IMPORTANT]
>Size / frames you can generate is dependent on available VRAM. Start low. Most likely fail point is VAE, after inference.
>Results saved to standard *Outputs* folder. Filename is *LTXvideo_**n**.mp4* where n resets to 0 at start of session.

>[!NOTE]
>known issue: result does not display in **Generated Output** section.
