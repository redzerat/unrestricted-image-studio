"""FLUX.2 [klein] 4B photoreal image generation service on Modal.

Endpoints (FastAPI, header auth: x-auth-token):
  POST /generate {prompt, width, height, seed} -> {"id": <function-call id>}
  GET  /result/<id> -> {"status": "pending"} | {"status": "done", "image_b64": ..., "seconds": ...} | {"status": "error", "error": ...}
  GET  /health -> {"ok": true}
"""
import io
import os
import time

import modal

APP_NAME = "flux-studio"
# Hardened fork (redzerat, 2026-11-25): Hugging Face revisions pinned to the
# commit SHAs read from the HF API on 2026-11-25. Update deliberately, never float.
REV_FLUX = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
REV_FLUX_ENCODER = "531dfe4c32ddcc6ec3aa1e146da905780281caf9"
REV_RV = "9a857a696b9aabbf509073e0aa55ec8200b6ef7d"
REV_VAE = "31f26fdeee1355a5c34592e401dd41e45d25a493"
REV_XL = "368374c797ebe7366781ce57c0ad92102ad69dd3"
REV_CHROMA = "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
REV_PONY = "def953b220844d4e1cee8bb8fbd8d56a1dfbabb1"
REV_FACEID = "43907e6f44d079bf1a9102d9a6e56aef7a219bae"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
# Second engine: Realistic Vision V6.0 B1 (SD1.5 community fine-tune).
# Uncensored by training, follows adult glamour prompts FLUX klein cannot.
RV_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
RV_NEGATIVE = (
    "deformed, distorted, disfigured, blurry, lowres, bad anatomy, bad hands, "
    "extra fingers, missing fingers, worst quality, low quality, cartoon, "
    "anime, painting, illustration, 3d render, watermark, text, logo, "
    "signature, caption, two people, multiple people, couple, group of "
    "people, second person, extra person, cloned face, duplicate person"
)
# Third engine: Lustify v2.0 (SDXL, CoyoteMix). RV cannot render two-person
# scenes at all (10/10 solo collapse, probe-verified 2026-11-13); Lustify is
# trained for explicit couples and draws both partners coherently.
XL_MODEL_ID = "TheImposterImposters/LUSTIFY-v2.0"
XL_NEGATIVE = (
    "deformed, distorted, blurry, lowres, bad anatomy, extra limbs, "
    "missing limbs, fused bodies, worst quality, cartoon, anime, painting, "
    "watermark, text, logo, signature, caption"
)
# Fourth engine: Chroma1-HD (FLUX.1-schnell derivative, Apache 2.0,
# uncensored by design, real CFG + negative prompts). Modern DiT stack:
# finer skin, hands and faces than the SDXL generation; slower per image.
CHROMA_ID = "lodestones/Chroma1-HD"
CHROMA_NEGATIVE = (
    "deformed, distorted, blurry, lowres, bad anatomy, bad hands, "
    "extra fingers, missing fingers, worst quality, cartoon, anime, "
    "painting, illustration, 3d render, watermark, text, logo, "
    "exaggerated anatomy, oversized genitals, deformed genitals, "
    "cartoonish anatomy, distorted proportions"
)
# Fifth engine: Pony Realism v2.1 MainVAE (SDXL; Pony Diffusion V6 realism
# merge, single-file checkpoint). Pony's training carries the strongest
# explicit-content knowledge of any engine here; its prompts want the
# score-tag prefix, which the caller adds.
PONY_ID = "LyliaEngine/ponyRealism_v21MainVAE"
PONY_FILE = "ponyRealism_v21MainVAE.safetensors"
PONY_NEGATIVE = (
    "score_4, score_5, score_6, lowres, bad anatomy, bad hands, extra "
    "fingers, missing fingers, deformed, distorted, blurry, worst quality, "
    "cartoon, anime, painting, 3d render, watermark, text, logo, signature"
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("flux-studio-models", create_if_missing=True)
auth_secret = modal.Secret.from_name("flux-studio-auth")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "diffusers>=0.36",
        "transformers",
        "accelerate",
        "sentencepiece",
        "pillow",
        "huggingface_hub",
        # FaceID stack (insightface pulls opencv, hence the libGL apt deps).
        "insightface",
        "onnxruntime",
        "numpy<2",
    )
    .env(
        {
            "HF_HOME": "/models/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "INSIGHTFACE_HOME": "/models/insightface",
            # Required after the encoder swap: without expandable segments the
            # allocator OOMs on T4 at 1024px (probe3-verified fix, peak 12.1 GiB).
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Bump to force an image rebuild (2026-11-12: deployed containers
            # hung silently in RV load while a fresh probe image ran clean).
            "IMAGE_REV": "2026-11-13-faceid",
        }
    )
)

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]"
)


@app.cls(
    # A10G (24GB): T4's ~14.6GB usable VRAM OOMs at 1024x1536 / 1536x1024
    # even with expandable_segments (user-hit 2026-11-12 21:16, log-verified).
    gpu="A10G",
    image=gpu_image,
    volumes={"/models": volume},
    scaledown_window=120,
    timeout=1200,
)
class Generator:
    @modal.enter()
    def load(self):
        # All engines load lazily on first use: an RV-only cold start must
        # not pay FLUX's ~16GB load (user-hit 2026-11-12 22:46, 3 RV jobs
        # queued ~10 min behind a cold FLUX load).
        self.pipe = None
        self.pipe_rv = None
        self.pipe_xl = None
        self.pipe_chroma = None
        self.pipe_pony = None
        self.faceid_embeds = {}
        self.xl_face_on = False

    def _load_flux(self):
        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import hf_hub_download

        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, revision=REV_FLUX
        )
        # Uncensored (abliterated) Qwen3 text encoder replaces the stock
        # safety-aligned one. Probe-verified 2026-11-12: its state dict loads
        # with 0 missing / 0 unexpected keys and normal-prompt quality holds.
        p = hf_hub_download(
            "PinoCookie/Flux.2-klein-4B-abliterated-text-encoder",
            "text_encoder.pt",
            revision=REV_FLUX_ENCODER,
            local_dir="/models/abliterated",
        )
        sd = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        res = self.pipe.text_encoder.load_state_dict(sd, strict=False)
        assert not res.missing_keys and not res.unexpected_keys, (
            res.missing_keys[:3],
            res.unexpected_keys[:3],
        )
        del sd
        self.pipe.enable_model_cpu_offload()
        volume.commit()

    def _load_xl(self):
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        self.pipe_xl = StableDiffusionXLPipeline.from_pretrained(
            XL_MODEL_ID, torch_dtype=torch.float16, revision=REV_XL
        )
        self.pipe_xl.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe_xl.scheduler.config, use_karras_sigmas=True
        )
        volume.commit()

    def _ensure_faceid(self, face: str = "face1"):
        """Load the FaceID adapter on the XL pipe and cache the reference
        face embedding per face id (probe8-verified 2026-11-13: the pipeline
        wants the CFG-stacked tensor [zero; face] in fp16, [2, 1, 512])."""
        import numpy as np
        import torch
        from PIL import Image

        if face not in self.faceid_embeds:
            from insightface.app import FaceAnalysis

            path = (
                "/models/ref/face-ref.jpg"
                if face == "face1"
                else f"/models/ref/{face}.jpg"
            )
            fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            fa.prepare(ctx_id=-1, det_size=(640, 640))
            ref = Image.open(path).convert("RGB")
            faces = fa.get(np.array(ref))
            assert faces, f"no face detected in reference {path}"
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            f1 = torch.tensor(
                faces[0].normed_embedding, dtype=torch.float16
            ).reshape(1, 1, -1)
            self.faceid_embeds[face] = torch.cat(
                [torch.zeros_like(f1), f1], dim=0
            )
            volume.commit()
        if not self.xl_face_on:
            self.pipe_xl.load_ip_adapter(
                "h94/IP-Adapter-FaceID",
                revision=REV_FACEID,
                subfolder="",
                weight_name="ip-adapter-faceid_sdxl.bin",
                image_encoder_folder=None,
            )
            self.pipe_xl.set_ip_adapter_scale(0.85)
            self.xl_face_on = True

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        model: str = "flux",
        face_ref: bool = False,
        face: str = "",
        steps: int = 0,
        guidance: float = 0.0,
    ) -> dict:
        if face_ref and not face:
            face = "face1"
        import torch

        t0 = time.time()
        if model == "rv":
            if self.pipe_rv is None:
                from diffusers import (
                    AutoencoderKL,
                    EulerAncestralDiscreteScheduler,
                    StableDiffusionPipeline,
                )
                from huggingface_hub import hf_hub_download

                # The diffusers-format repo garbles output; the single-file
                # checkpoint + sd-vae-ft-mse + Euler a is the verified stack
                # (probe4, 2026-11-12).
                ckpt = hf_hub_download(
                    RV_MODEL_ID,
                    "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
                    revision=REV_RV,
                    local_dir="/models/rv",
                )
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16,
                    revision=REV_VAE
                )
                self.pipe_rv = StableDiffusionPipeline.from_single_file(
                    ckpt, vae=vae, torch_dtype=torch.float16, safety_checker=None
                )
                self.pipe_rv.scheduler = (
                    EulerAncestralDiscreteScheduler.from_config(
                        self.pipe_rv.scheduler.config
                    )
                )
                volume.commit()
            self.pipe_rv.to("cuda")
            # SD1.5 native range: cap the long side at 1024, keep aspect, /8.
            w, h = int(width), int(height)
            scale = min(1.0, 1024 / max(w, h))
            w = max(256, int(w * scale) // 8 * 8)
            h = max(256, int(h * scale) // 8 * 8)
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            out = self.pipe_rv(
                prompt=prompt,
                negative_prompt=RV_NEGATIVE,
                width=w,
                height=h,
                num_inference_steps=20,
                guidance_scale=7.0,
                generator=g,
            ).images[0]
        elif model == "lustify":
            if self.pipe_xl is None:
                self._load_xl()
            self.pipe_xl.to("cuda")
            # SDXL native range: aim for ~1024px on the long side.
            w, h = int(width), int(height)
            scale = 1024 / max(w, h)
            w = max(512, int(w * scale) // 8 * 8)
            h = max(512, int(h * scale) // 8 * 8)
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            kw = {}
            if face:
                self._ensure_faceid(face)
                kw["ip_adapter_image_embeds"] = [
                    self.faceid_embeds[face].to("cuda")
                ]
            elif self.xl_face_on:
                self.pipe_xl.unload_ip_adapter()
                self.xl_face_on = False
            out = self.pipe_xl(
                prompt=prompt,
                negative_prompt=XL_NEGATIVE,
                width=w,
                height=h,
                num_inference_steps=30,
                guidance_scale=6.0,
                generator=g,
                **kw,
            ).images[0]
        elif model == "chroma":
            # Free VRAM for Chroma's offload peaks.
            for p in (self.pipe_rv, self.pipe_xl, self.pipe_pony):
                if p is not None:
                    p.to("cpu")
            torch.cuda.empty_cache()
            if self.pipe_chroma is None:
                from diffusers import ChromaPipeline

                self.pipe_chroma = ChromaPipeline.from_pretrained(
                    CHROMA_ID, torch_dtype=torch.bfloat16, revision=REV_CHROMA
                )
                self.pipe_chroma.enable_model_cpu_offload()
                volume.commit()
            w, h = int(width), int(height)
            scale = min(1.0, 1024 / max(w, h))
            w = max(512, int(w * scale) // 16 * 16)
            h = max(512, int(h * scale) // 16 * 16)
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            out = self.pipe_chroma(
                prompt=prompt,
                negative_prompt=CHROMA_NEGATIVE,
                width=w,
                height=h,
                num_inference_steps=int(steps) if steps else 40,
                guidance_scale=float(guidance) if guidance else 2.5,
                generator=g,
            ).images[0]
        elif model == "pony":
            if self.pipe_pony is None:
                from diffusers import (
                    DPMSolverMultistepScheduler,
                    StableDiffusionXLPipeline,
                )
                from huggingface_hub import hf_hub_download

                ckpt = hf_hub_download(
                    PONY_ID, PONY_FILE, local_dir="/models/pony",
                    revision=REV_PONY
                )
                self.pipe_pony = StableDiffusionXLPipeline.from_single_file(
                    ckpt, torch_dtype=torch.float16
                )
                self.pipe_pony.scheduler = (
                    DPMSolverMultistepScheduler.from_config(
                        self.pipe_pony.scheduler.config, use_karras_sigmas=True
                    )
                )
                volume.commit()
            self.pipe_pony.to("cuda")
            # SDXL native range: aim for ~1024px on the long side.
            w, h = int(width), int(height)
            scale = 1024 / max(w, h)
            w = max(512, int(w * scale) // 8 * 8)
            h = max(512, int(h * scale) // 8 * 8)
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            out = self.pipe_pony(
                prompt=prompt,
                negative_prompt=PONY_NEGATIVE,
                width=w,
                height=h,
                num_inference_steps=30,
                guidance_scale=6.0,
                generator=g,
            ).images[0]
        else:
            # Keep the A10G free for FLUX's offload peaks.
            if self.pipe_rv is not None:
                self.pipe_rv.to("cpu")
                torch.cuda.empty_cache()
            if self.pipe_xl is not None:
                self.pipe_xl.to("cpu")
                torch.cuda.empty_cache()
            if self.pipe_pony is not None:
                self.pipe_pony.to("cpu")
                torch.cuda.empty_cache()
            if self.pipe is None:
                self._load_flux()
            g = torch.Generator(device="cuda").manual_seed(int(seed))
            out = self.pipe(
                prompt=prompt,
                width=int(width),
                height=int(height),
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=g,
            ).images[0]
        buf = io.BytesIO()
        out.convert("RGB").save(buf, format="JPEG", quality=92)
        return {"jpeg": buf.getvalue(), "seconds": round(time.time() - t0, 1)}


@app.function(image=cpu_image, secrets=[auth_secret])
@modal.asgi_app()
def web():
    import base64
    import secrets

    from fastapi import FastAPI, Header, HTTPException, Request

    api = FastAPI()

    def check(token):
        # Hardened: fail closed when the secret is missing/empty (upstream
        # compared empty header == empty env and let the request through),
        # and compare in constant time.
        expected = os.environ.get("AUTH_TOKEN", "")
        if not expected:
            raise HTTPException(status_code=503, detail="auth not configured")
        if not secrets.compare_digest(token or "", expected):
            raise HTTPException(status_code=401, detail="bad token")

    def translate_zh(text: str):
        """Chinese -> English for the CLIP-based engines. Google gtx first,
        MyMemory as fallback; returns None when both fail (caller then keeps
        the original prompt)."""
        import json as _json
        import urllib.parse
        import urllib.request

        q = urllib.parse.quote(text)
        try:
            req = urllib.request.Request(
                "https://translate.googleapis.com/translate_a/single"
                f"?client=gtx&sl=zh-CN&tl=en&dt=t&q={q}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read().decode("utf-8"))
            out = "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()
            if out:
                return out
        except Exception:
            pass
        try:
            req = urllib.request.Request(
                "https://api.mymemory.translated.net/get"
                f"?q={q}&langpair=zh-CN|en",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read().decode("utf-8"))
            out = str(data["responseData"]["translatedText"]).strip()
            if out:
                return out
        except Exception:
            pass
        return None

    @api.get("/health")
    async def health():
        return {"ok": True}

    @api.post("/generate")
    async def generate(req: Request, x_auth_token: str = Header(default="")):
        check(x_auth_token)
        body = await req.json()
        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        if len(prompt) > 2000:
            raise HTTPException(status_code=400, detail="prompt too long (max 2000 chars)")
        prompt_in = prompt
        translated = False
        translation_disabled = False
        if any("一" <= ch <= "鿿" for ch in prompt):
            # Hardened: upstream sent Chinese prompts in plaintext to Google
            # and MyMemory by default. Translation is now OFF unless the
            # operator explicitly sets TRANSLATE_ZH=1 on the web function.
            if os.environ.get("TRANSLATE_ZH", "") == "1":
                t = translate_zh(prompt)
                if t:
                    prompt = t
                    translated = True
            else:
                translation_disabled = True
        width = max(256, min(1536, int(body.get("width", 1024))))
        height = max(256, min(1536, int(body.get("height", 1024))))
        width -= width % 16
        height -= height % 16
        seed = int(body.get("seed", 0)) or int(time.time()) % 1_000_000
        model = str(body.get("model", "flux")).lower()
        if model not in ("flux", "rv", "lustify", "chroma", "pony"):
            model = "flux"
        face = str(body.get("face", "")).lower()
        if body.get("face_ref") and not face:
            face = "face1"
        if model != "lustify" or face not in ("face1", "face2"):
            face = ""
        steps = max(0, min(60, int(body.get("steps", 0) or 0)))
        guidance = max(0.0, min(8.0, float(body.get("guidance", 0) or 0)))
        call = Generator().generate.spawn(
            prompt, width, height, seed, model, False, face, steps, guidance
        )
        return {
            "id": call.object_id,
            "seed": seed,
            "width": width,
            "height": height,
            "model": model,
            "face": face,
            "prompt_in": prompt_in,
            "prompt_used": prompt,
            "translated": translated,
            "translation_disabled": translation_disabled,
        }

    @api.get("/result/{call_id}")
    async def result(call_id: str, x_auth_token: str = Header(default="")):
        check(x_auth_token)
        from modal.functions import FunctionCall

        fc = FunctionCall.from_id(call_id)
        try:
            out = fc.get(timeout=0)
        except TimeoutError:
            return {"status": "pending"}
        except Exception as e:  # generation raised inside the GPU function
            return {"status": "error", "error": str(e)[:500]}
        return {
            "status": "done",
            "seconds": out["seconds"],
            "image_b64": base64.b64encode(out["jpeg"]).decode(),
        }

    return api
