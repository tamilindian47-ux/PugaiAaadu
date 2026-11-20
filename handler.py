import os
import glob
import json
import time
import base64
import traceback
import urllib.request
import urllib.error
import subprocess
import runpod

COMFY_DIR = "/comfyui_latest"
COMFY_HOST = "127.0.0.1:8188"
INPUT_DIR = f"{COMFY_DIR}/input"
OUTPUT_DIR = f"{COMFY_DIR}/output"
COMFY_LOG = "/tmp/comfy.log"
BOOT_ERROR = None
COMFY_READY = False


def boot_comfy():
    global BOOT_ERROR, COMFY_READY
    if COMFY_READY:
        return
    try:
        if not os.path.exists("/usr/lib/x86_64-linux-gnu/libcuda.so"):
            matches = glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so.*") + glob.glob("/usr/local/cuda/lib64/stubs/libcuda.so*")
            first_match = next(iter(matches), None)
            if first_match:
                os.symlink(first_match, "/usr/lib/x86_64-linux-gnu/libcuda.so")
                subprocess.run(["ldconfig"], check=False)

        env = os.environ.copy()
        env["COMFY_KITCHEN_DISABLE_TRITON"] = "1"
        env["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64/stubs:" + env.get("LIBRARY_PATH", "")

        log_fh = open(COMFY_LOG, "w")
        subprocess.Popen(
            ["/opt/venv/bin/python", f"{COMFY_DIR}/main.py", "--listen", "127.0.0.1", "--port", "8188"],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env
        )

        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats", timeout=2)
                COMFY_READY = True
                return
            except Exception:
                time.sleep(1)

        with open(COMFY_LOG, "r", errors="replace") as f:
            BOOT_ERROR = f"ComfyUI failed to start within 90s. Log:\n{f.read()[-2500:]}"
    except Exception:
        BOOT_ERROR = traceback.format_exc()


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
    raw = urllib.request.urlopen(f"http://{COMFY_HOST}/object_info", timeout=10).read()
    info = json.loads(raw) or {}

    text_cond_cls = None
    cond_cls = None
    for cls_name, meta in info.items():
        meta = meta or {}
        disp = meta.get("display_name") or ""
        if "Load Text Conditioning" in disp or ("Viggle" in cls_name and "Text" in cls_name):
            text_cond_cls = cls_name
        if disp == "Viggle Animate Conditioning" or ("Viggle" in cls_name and "Conditioning" in cls_name and "Window" not in cls_name):
            cond_cls = cls_name

    if not text_cond_cls or not cond_cls:
        with open(COMFY_LOG, "r", errors="replace") as f:
            tail = f.read()[-3000:]
        viggle_keys = [k for k in info.keys() if "viggle" in k.lower() or "minimax" in k.lower()]
        raise RuntimeError(f"Viggle nodes missing. Matching: {viggle_keys}\nComfyUI Log:\n{tail}")

    tc_meta = info.get(text_cond_cls) or {}
    tc_input = tc_meta.get("input") or {}
    tc_req = tc_input.get("required") or {}
    tc_key = next(iter(tc_req.keys()))

    cond_meta = info.get(cond_cls) or {}
    cond_input = cond_meta.get("input") or {}
    cond_req = cond_input.get("required") or {}
    cond_opt = cond_input.get("optional") or {}

    cond_inputs = {}
    for k in list(cond_req.keys()) + list(cond_opt.keys()):
        kl = k.lower()
        if "pos" in kl or "cond" in kl:
            cond_inputs[k] = ["5", 0]
        elif "vae" in kl:
            cond_inputs[k] = ["4", 0]
        elif "ref" in kl or "img" in kl or "image" in kl:
            cond_inputs[k] = ["6", 0]
        elif "vid" in kl or "drive" in kl or "driving" in kl:
            cond_inputs[k] = ["7", 0]
        elif "len" in kl or "frame" in kl:
            cond_inputs[k] = frame_cap
        elif "width" in kl:
            cond_inputs[k] = 480
        elif "height" in kl:
            cond_inputs[k] = 832

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
            "class_type": text_cond_cls,
            "inputs": {
                tc_key: "fixed_embed_fwd_anyframe.safetensors"
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
            "class_type": cond_cls,
            "inputs": cond_inputs
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
    boot_comfy()
    if BOOT_ERROR:
        return {"error": BOOT_ERROR}

    try:
        job_input = job.get("input") or {}
        ref_input = job_input.get("ref_image")
        vid_input = job_input.get("driving_video")
        frame_cap = int(job_input.get("frame_cap") or 124)
        seed = int(job_input.get("seed") or 42)

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
        try:
            resp = json.loads(urllib.request.urlopen(req).read()) or {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return {"error": f"ComfyUI 400 response: {err_body}"}

        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            return {"error": f"No prompt_id returned: {resp}"}

        deadline = time.time() + 600
        while time.time() < deadline:
            time.sleep(2)
            hist_req = urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}")
            history = json.loads(hist_req.read()) or {}
            if prompt_id in history:
                entry = history[prompt_id]
                status = (entry.get("status") or {}).get("status_str")
                if status == "error":
                    return {"error": f"ComfyUI execution failed: {entry.get('status')}"}

                outputs = entry.get("outputs") or {}
                for node_out in outputs.values():
                    for key in ("gifs", "videos"):
                        for item in node_out.get(key) or []:
                            fname = item.get("filename")
                            sub = item.get("subfolder") or ""
                            fpath = os.path.join(OUTPUT_DIR, sub, fname)
                            if os.path.exists(fpath):
                                with open(fpath, "rb") as vf:
                                    b64 = base64.b64encode(vf.read()).decode("utf-8")
                                return {"video_base64": b64}
                return {"error": f"Completed but no video output found: {outputs}"}

        return {"error": "Timed out waiting for ComfyUI render."}
    except Exception:
        return {"error": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
