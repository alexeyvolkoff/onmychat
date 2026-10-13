#!/usr/bin/env python3
import urllib.request, json, time, sys, os, subprocess

prompt_text = "cinematic fashion portrait, beautiful young woman, stylish trench coat, rainy Paris street at night, neon reflections on wet asphalt, soft bokeh lights, hyperrealistic, sharp focus, 8k photography"
if len(sys.argv) > 1 and sys.argv[1].strip():
    prompt_text = sys.argv[1].strip()

workflow = {
  "10": {
    "inputs": {
      "unet_name": "CyberRealistic_Krea2_V4.0_BF16.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader"
  },
  "11": {
    "inputs": {
      "clip_name": "qwen3vl_4b_bf16.safetensors",
      "type": "krea2"
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
  "13": {
    "inputs": {
      "conditioning": ["6", 0]
    },
    "class_type": "ConditioningZeroOut"
  },
  "5": {
    "inputs": {
      "width": 896,
      "height": 1152,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage"
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
      "negative": ["13", 0],
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
      "filename_prefix": "Krea2/krea2",
      "images": ["8", 0]
    },
    "class_type": "SaveImage"
  }
}

req_data = json.dumps({"prompt": workflow}).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=req_data, headers={"Content-Type": "application/json"})

print(f"🎨 Отправка запроса в ComfyUI...")
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
            print(f"✅ Генерация завершена за {elapsed:.2f} сек!")
            subprocess.run(["chown", "-R", "omduser:omduser", "/home/omduser/Data/Krea2"], check=False)
            subprocess.run(["chmod", "-R", "777", "/home/omduser/Data/Krea2"], check=False)
            break

print("\n📁 Файлы в /home/omduser/Data/Krea2/:")
print(subprocess.getoutput("ls -lh /home/omduser/Data/Krea2"))
