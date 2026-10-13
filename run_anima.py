#!/usr/bin/env python3
import urllib.request, json, time, sys, os, subprocess

default_prompt = "masterpiece, best quality, 1girl, green eyes, wavy auburn hair, freckles, white summer dress, laughing happily, sunny outdoor street, vibrant anime colors, clean 2D cel shading, Kyoto anime aesthetic, high detail"
prompt_text = default_prompt
if len(sys.argv) > 1 and sys.argv[1].strip():
    prompt_text = sys.argv[1].strip()

workflow = {
  "10": {
    "inputs": {
      "unet_name": "anima-turbo-v1.1.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader"
  },
  "11": {
    "inputs": {
      "clip_name": "qwen_3_06b_base.safetensors",
      "type": "stable_diffusion",
      "device": "default"
    },
    "class_type": "CLIPLoader"
  },
  "12": {
    "inputs": {
      "vae_name": "qwen_image_vae.safetensors"
    },
    "class_type": "VAELoader"
  },
  "6": {
    "inputs": {
      "text": prompt_text,
      "clip": ["11", 0]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, photo, realistic, 3d render",
      "clip": ["11", 0]
    },
    "class_type": "CLIPTextEncode"
  },
  "5": {
    "inputs": {
      "width": 896,
      "height": 1152,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage"
  },
  "3": {
    "inputs": {
      "seed": int(time.time() * 1000) % 2147483647,
      "steps": 8,
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0,
      "model": ["10", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    },
    "class_type": "KSampler"
  },
  "8": {
    "inputs": {
      "samples": ["3", 0],
      "vae": ["12", 0]
    },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "Anima/anima",
      "images": ["8", 0]
    },
    "class_type": "SaveImage"
  }
}

req_data = json.dumps({"prompt": workflow}).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=req_data, headers={"Content-Type": "application/json"})

print(f"🎨 Отправка запроса Anima в ComfyUI...")
print(f"Промпт: {prompt_text}")
start = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read())
    prompt_id = res["prompt_id"]

while True:
    time.sleep(0.5)
    with urllib.request.urlopen("http://127.0.0.1:8188/history/" + prompt_id) as h_resp:
        h = json.loads(h_resp.read())
        if prompt_id in h:
            elapsed = time.time() - start
            print(f"✅ Генерация Anima завершена за {elapsed:.2f} сек!")
            subprocess.run(["chown", "-R", "omduser:omduser", "/content/ComfyUI/output/Anima"], check=False)
            subprocess.run(["chmod", "-R", "777", "/content/ComfyUI/output/Anima"], check=False)
            break

print("\n📁 Файлы в /home/omduser/Data/Anima/:")
print(subprocess.getoutput("ls -lh /home/omduser/Data/Anima/"))
