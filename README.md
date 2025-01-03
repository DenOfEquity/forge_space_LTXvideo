## Forge2 Spaces implementation of LTX-video ##
New Forge only.

* https://huggingface.co/Lightricks
* https://github.com/Lightricks/LTX-Video
* https://huggingface.co/spaces/Lightricks/LTX-Video-Playground

uses *diffusers*

will download ~10.8GB models

>[!NOTE]
>Install via *Extensions* tab; *Install from URL* sub-tab; use URL of this repo.
>
>Requirements:
>`diffusers>=0.32.0`
>
>`imageio-ffmpeg`

>[!TIP]
>`noUnload` toggle keeps models in memory
>
>`lowVRAM` one-way switch enables sequential model offloading - much lower VRAM usage, but slower: usually better than hitting shared memory.

>[!IMPORTANT]
>Size / frames you can generate is dependent on available VRAM. Start low. Most likely fail point is VAE, after inference.
>
>Results saved to standard *Outputs* folder. Filename is *LTXvideo_**datetime**.mp4*.

>[!NOTE]
>Also saves a text file with same name (*LTXvideo_**datetime**.txt*), containing generation parameters.
