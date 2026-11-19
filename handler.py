import os
import json
import time
import base64
import urllib.request
import subprocess
import runpod

COMFY_HOST = "127.0.0.1:8188"
INPUT_DIR = "/comfyui/input"
OUTPUT_DIR = "/comfyui/output"

subprocess.Popen(
    ["python3", "/comfyui/main.py", "--listen", "127.0.0.1", "--port", "8188"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

def wait_for_comfy():
    for _ in range(90):
        try:
            urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI server failed to start.")

wait_for_comfy()

def save_input_file(data_or_url, filename):
    filepath = os.path.join(INPUT_DIR, filename)
    if data_or_url.startswith("http://") or data_or_url.startswith("https://"):
        urllib.request.urlretrieve(data_or_url, filepath)
    else:
        _, _, b64_data = data_or_url.rpartition(",")
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(b64_data))
    return filename

def build_viggle_workflow(ref_image_name, driving_video_name, frame_cap=124, seed=42):
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "viggle_animate_dmd_lora_r64.safetensors",
                "strength_model": 1.0
            }
        },
        "3": {
            "class_type": "ModelSamplingMiniMaxH3",
            "inputs": {
                "model": ["2", 0],
                "video_shift": 3.0,
                "audio_shift": 3.0
            }
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "minimax_h3_video_vae_fp16.safetensors"
            }
        },
        "5": {
            "class_type": "ViggleLoadTextCond",
            "inputs": {
                "text_cond_name": "fixed_embed_fwd_anyframe.safetensors"
            }
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {
                "image": ref_image_name
            }
        },
        "7": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": driving_video_name,
                "force_rate": 24,
                "force_size": "Custom",
                "custom_width": 480,
                "custom_height": 832,
                "frame_load_cap": frame_cap,
                "skip_first_frames": 0,
                "select_every_nth": 1
            }
        },
        "8": {
            "class_type": "ViggleAnimateConditioning",
            "inputs": {
                "positive": ["5", 0],
                "vae": ["4", 0],
                "ref_image": ["6", 0],
                "driving_video": ["7", 0],
                "length": frame_cap
            }
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["3", 0],
                "conditioning": ["8", 0]
            }
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed
            }
        },
        "11": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        "12": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.75, 0.5, 0.25, 0.0"
            }
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["9", 0],
                "sampler": ["11", 0],
                "sigmas": ["12", 0],
                "latent_image": ["8", 1]
            }
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["4", 0]
            }
        },
        "15": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["14", 0],
                "audio": ["7", 2],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "viggle_out",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True
            }
        }
    }

def handler(job):
    job_input = job.get("input", {})
    ref_input = job_input.get("ref_image")
    vid_input = job_input.get("driving_video")
    frame_cap = int(job_input.get("frame_cap", 124))
    seed = int(job_input.get("seed", 42))

    if not ref_input or not vid_input:
        return {"error": "Both ref_image and driving_video (base64 or URL) are required."}

    os.makedirs(INPUT_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    ref_name = save_input_file(ref_input, f"ref_{ts}.png")
    vid_name = save_input_file(vid_input, f"drv_{ts}.mp4")

    workflow = build_viggle_workflow(ref_name, vid_name, frame_cap=frame_cap, seed=seed)
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{COMFY_HOST}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    prompt_id = resp.get("prompt_id")

    while True:
        time.sleep(2)
        hist_req = urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}")
        history = json.loads(hist_req.read())
        if prompt_id in history:
            outputs = history.get(prompt_id, {}).get("outputs", {})
            gifs = outputs.get("15", {}).get("gifs", [])
            if gifs:
                video_info = next(iter(gifs))
                out_path = os.path.join(OUTPUT_DIR, video_info.get("filename"))
                with open(out_path, "rb") as vf:
                    out_b64 = base64.b64encode(vf.read()).decode("utf-8")
                return {
                    "video_base64": out_b64,
                    "filename": video_info.get("filename")
                }
            break

    return {"error": "Generation finished without producing an output video."}

runpod.serverless.start({"handler": handler})
