FROM runpod/worker-comfyui:5.2.0-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    gcc \
    libc6-dev \
    python3-dev \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

RUN for d in ComfyUI-Viggle-Animate-H3 ComfyUI-KJNodes ComfyUI-VideoHelperSuite; do \
        if [ -f "$d/requirements.txt" ]; then \
            /opt/venv/bin/pip install --no-cache-dir -r "$d/requirements.txt"; \
        fi; \
    done

RUN mkdir -p /comfyui/models/diffusion_models \
             /comfyui/models/loras \
             /comfyui/models/text_cond \
             /comfyui/models/vae

RUN wget -q "https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors" \
      -O /comfyui/models/diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors && \
    wget -q "https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/loras/viggle_animate_dmd_lora_r64.safetensors" \
      -O /comfyui/models/loras/viggle_animate_dmd_lora_r64.safetensors && \
    wget -q "https://huggingface.co/drbaph/Viggle-Animate-ComfyUI/resolve/main/text_cond/fixed_embed_fwd_anyframe.safetensors" \
      -O /comfyui/models/text_cond/fixed_embed_fwd_anyframe.safetensors && \
    wget -q "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
      -O /comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /comfyui_latest && \
    rm -rf /comfyui_latest/models && \
    ln -s /comfyui/models /comfyui_latest/models && \
    for node_dir in ComfyUI-Viggle-Animate-H3 ComfyUI-KJNodes ComfyUI-VideoHelperSuite; do \
        ln -s "/comfyui/custom_nodes/${node_dir}" "/comfyui_latest/custom_nodes/${node_dir}"; \
    done && \
    /opt/venv/bin/pip install --no-cache-dir -r /comfyui_latest/requirements.txt

WORKDIR /
COPY handler.py /handler.py

CMD ["bash", "-c", "wget -q https://raw.githubusercontent.com/tamilindian47-ux/PugaiAaadu/main/handler.py -O /handler.py.new && mv /handler.py.new /handler.py || true; /opt/venv/bin/python -u /handler.py"]
