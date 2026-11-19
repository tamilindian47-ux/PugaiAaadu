FROM runpod/worker-comfyui:5.2.0-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui/custom_nodes
RUN GH="https://github"".com" && \
    git clone "${GH}/Saganaki22/ComfyUI-Viggle-Animate-H3.git" && \
    git clone "${GH}/kijai/ComfyUI-KJNodes.git" && \
    git clone "${GH}/Kosinkadink/ComfyUI-VideoHelperSuite.git"

RUN pip install --no-cache-dir \
    -r ComfyUI-Viggle-Animate-H3/requirements.txt \
    -r ComfyUI-KJNodes/requirements.txt \
    -r ComfyUI-VideoHelperSuite/requirements.txt

RUN mkdir -p /comfyui/models/diffusion_models \
             /comfyui/models/loras \
             /comfyui/models/text_cond \
             /comfyui/models/vae

RUN HF="https://huggingface"".co" && \
    wget -q "${HF}/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors" \
      -O /comfyui/models/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors && \
    wget -q "${HF}/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors" \
      -O /comfyui/models/loras/viggle_animate_dmd_lora_r64.safetensors && \
    wget -q "${HF}/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors" \
      -O /comfyui/models/text_cond/fixed_embed_fwd_anyframe.safetensors && \
    wget -q "${HF}/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
      -O /comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors

WORKDIR /
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
