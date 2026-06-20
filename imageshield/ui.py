"""
Design User Interface (Web-based)
1. The model (baseline of either EditShield or PhotoGuard) has input of an image, and an output of a protected version of the image.
2. The web-based interface should be made user friendly, such that anyone is able to use it easily (it should not involve
any code running inside the terminal)
"""

# Import necessary packages
import base64
import os
import time
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image as _PILImage

from .protection import (
    EPS_IP2P,
    EPS_SD,
    ProtectionCancelled,
    ProtectionService,
    ProtectionSettings,
    device_summary,
)
from .resources import resource_path, user_data_dir

SERVICE = ProtectionService()
OUTPUT_DIR = user_data_dir() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_OUTPUT_AGE_SECONDS = 24 * 60 * 60
BG_IMAGE_PATH = Path(__file__).with_name("shield_bg.png")
DEMO_IMAGE_DIR = resource_path("imageshield", "demo_image")

# EPS slider range displayed in the UI (values in units of 1/255)
_EPS_MIN_255 = 1
_EPS_MAX_255 = 32
_EPS_IP2P_255 = round(EPS_IP2P * 255)   # 4
_EPS_SD_255   = round(EPS_SD   * 255)   # 16


def _image_data_url(filename: str) -> str:
    path = DEMO_IMAGE_DIR / filename
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:{mime};base64,{encoded}"


def _demo_strength_gallery_html() -> str:
    examples = [
        ("Original", "0035_0.jpg", "Unprotected input"),
        ("Low", "protected-IP2P-512-100-0.0156.png", "IP2P, eps approx 4/255"),
        ("Medium", "protected-IP2P-512-20-0.03.png", "IP2P, eps approx 8/255"),
        ("High", "protected-IP2P-512-100-0.05.png", "IP2P, eps approx 13/255"),
    ]
    cards = []
    for label, filename, description in examples:
        src = _image_data_url(filename)
        if not src:
            continue
        cards.append(
            f"""
            <figure class="strength-demo-card">
              <img src="{src}" alt="{label} perturbation example">
              <figcaption>
                <strong>{label}</strong>
                <span>{description}</span>
              </figcaption>
            </figure>
            """
        )
    if len(cards) != len(examples):
        return ""
    return f"""
      <div class="strength-demo-grid">
        {''.join(cards)}
      </div>
      <p class="strength-demo-note">
        The examples show how stronger perturbation budgets can make protection more visible.
        Actual visibility depends on the image content, mode, resolution, step count, and display size.
      </p>
    """


def cleanup_old_outputs() -> None:
    """Remove generated files older than one day without blocking app startup."""
    cutoff = time.time() - MAX_OUTPUT_AGE_SECONDS
    for path in OUTPUT_DIR.glob("protected-*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def prereq_gpu(image):
    """
    Before running the model, detect whether or not the current PC has a GPU to see if the computations can be done successfully.
    Supports NVIDIA GPUs (Windows/Linux) and Apple Silicon MPS (macOS).
    """
    if image is None:
        raise gr.Error("Please upload an image first.")
    return gr.update(value=device_summary(SERVICE.device), visible=True)


def model(image, mode, resolution, eps_255, steps, progress=gr.Progress()):
    """
    The function is the root of the model function. The model acts as the image protector, given an image as an input, it will return the protected version
    of the image as an output.
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    # resolution == "Original" means None (keep source dimensions)
    resolution_int = None if resolution == "Original" else int(resolution)
    eps_float = float(eps_255) / 255.0

    settings = ProtectionSettings(
        resolution=resolution_int,
        eps=eps_float,
        steps=int(steps),
        mode=str(mode).lower(),
    )

    try:
        progress(0, desc="Loading the offline protection model")
        protected_image = SERVICE.protect(
            image,
            settings=settings,
            progress=lambda value, description: progress(value, desc=description),
        )
    except ProtectionCancelled:
        raise gr.Error("Protection stopped.")
    except Exception as exc:
        raise gr.Error(f"Protection failed: {exc}") from exc

    output_path = OUTPUT_DIR / f"protected-{uuid.uuid4().hex}.png"
    protected_image.save(output_path, format="PNG")
    image_file_name = Path(output_path).name
    return (
        protected_image,
        gr.update(label=image_file_name, value=str(output_path), visible=True),
    )


def stop_protection():
    SERVICE.cancel()
    return gr.update(
        value="Stopping protection after the current optimization step...",
        visible=True,
    )


def _preview_resize(image, resolution):
    """Resize + center-crop a PIL image to match model preprocessing (for display only)."""
    if image is None or resolution == "Original":
        return image
    size = int(resolution)
    w, h = image.size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_h, new_w = size, int(w * size / h)
    resized = image.resize((new_w, new_h), _PILImage.Resampling.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return resized.crop((left, top, left + size, top + size))


# ---------------------------------------------------------------------------
# Navigation bar — "About" link sits on the right
# ---------------------------------------------------------------------------
NAV_HTML = """
<div id="main-nav" style="
    display:flex; justify-content:space-between; align-items:center;
    padding:13px 26px;
    background:rgba(6,14,28,0.92);
    border-radius:18px;
    border:1px solid rgba(56,232,255,0.40);
    margin-bottom:20px;
    box-shadow:0 0 20px rgba(56,232,255,0.12),0 8px 30px rgba(0,0,0,0.55);
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    font-family:'Inter',system-ui,sans-serif;">
  <div style="display:flex;align-items:center;gap:12px;">
    <svg width="38" height="44" viewBox="-20 -23 40 47" xmlns="http://www.w3.org/2000/svg"
         style="flex-shrink:0;filter:drop-shadow(0 0 6px rgba(56,232,255,0.60));">
      <path d="M0-22 L17-15 L17-4 C17 9 9 14 0 17 C-9 14-17 9-17-4 L-17-15 Z"
            fill="rgba(10,20,55,0.95)" stroke="rgba(56,232,255,0.90)" stroke-width="1.4"/>
      <path d="M0-18 L13-12 L13-2 C13 6 7 10 0 13 C-7 10-13 6-13-2 L-13-12 Z"
            fill="none" stroke="rgba(56,232,255,0.28)" stroke-width="0.7"/>
      <ellipse cx="0" cy="-2" rx="11" ry="3.5" fill="none"
               stroke="rgba(56,232,255,0.30)" stroke-width="0.6" stroke-dasharray="4 4"/>
      <circle cx="0" cy="-2" r="8"
              fill="rgba(3,6,26,0.90)" stroke="rgba(56,232,255,0.75)" stroke-width="1.1"/>
      <circle cx="0" cy="-2" r="5"
              fill="rgba(100,28,158,0.65)" stroke="rgba(139,92,246,0.60)" stroke-width="0.8"/>
      <circle cx="0" cy="-2" r="2.5" fill="#38E8FF" opacity="0.95"/>
      <circle cx="-1.2" cy="-3.2" r="0.9" fill="rgba(255,255,255,0.88)"/>
    </svg>
    <div>
      <div style="font-size:17px;font-weight:800;color:#38E8FF;
                  letter-spacing:0.12em;text-transform:uppercase;
                  text-shadow:0 0 16px rgba(56,232,255,0.65);
                  line-height:1.2;">SafeShot</div>
      <div style="font-size:10px;font-weight:500;color:rgba(56,232,255,0.50);
                  letter-spacing:0.09em;text-transform:uppercase;line-height:1;">
        AI-Powered Image Protection
      </div>
    </div>
  </div>
  <div style="display:flex;gap:26px;align-items:center;">
    <a onclick="window.showPage('main')" id="nav-home"
       style="cursor:pointer;font-size:14px;font-weight:600;color:#C4D4E4;
              text-decoration:none;padding:5px 2px;
              border-bottom:2px solid transparent;letter-spacing:0.04em;"
       onmouseover="if(this.style.borderBottomColor!='rgb(56, 232, 255)'){this.style.color='#38E8FF';this.style.textShadow='0 0 10px rgba(56,232,255,0.5)';}"
       onmouseout="if(this.style.borderBottomColor!='rgb(56, 232, 255)'){this.style.color='#C4D4E4';this.style.textShadow='none';}">
      Home
    </a>
    <a onclick="window.showPage('guide')" id="nav-guide"
       style="cursor:pointer;font-size:14px;font-weight:600;color:#C4D4E4;
              text-decoration:none;padding:5px 2px;
              border-bottom:2px solid transparent;letter-spacing:0.04em;"
       onmouseover="if(this.style.borderBottomColor!='rgb(56, 232, 255)'){this.style.color='#38E8FF';this.style.textShadow='0 0 10px rgba(56,232,255,0.5)';}"
       onmouseout="if(this.style.borderBottomColor!='rgb(56, 232, 255)'){this.style.color='#C4D4E4';this.style.textShadow='none';}">
      User Guide
    </a>
    <a onclick="window.showPage('about')" id="nav-about"
       style="cursor:pointer;font-size:14px;font-weight:600;color:#C4D4E4;
              text-decoration:none;padding:5px 2px;
              border-bottom:2px solid transparent;letter-spacing:0.04em;"
       onmouseover="if(this.style.borderBottomColor!='rgb(56, 232, 255)'){this.style.color='#38E8FF';this.style.textShadow='0 0 10px rgba(56,232,255,0.5)';}"
       onmouseout="if(this.style.borderBottomColor!='rgb(56, 232, 255)'){this.style.color='#C4D4E4';this.style.textShadow='none';}">
      About
    </a>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Spotlight tour overlay — HTML skeleton only (Gradio strips <script> tags)
# ---------------------------------------------------------------------------
TOUR_HTML = """
<div id="tour-root">
  <svg id="tour-svg"
       style="display:none; position:fixed; top:0; left:0;
              width:100vw; height:100vh; z-index:10000; pointer-events:none;">
    <defs>
      <mask id="tour-mask">
        <rect width="100%" height="100%" fill="white"/>
        <rect id="tour-hole" rx="10" ry="10" fill="black"/>
      </mask>
    </defs>
    <rect width="100%" height="100%" fill="rgba(2,8,20,0.75)" mask="url(#tour-mask)"/>
  </svg>
  <div id="tour-blocker"
       style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; z-index:9999;"></div>
  <div id="tour-card"
       style="display:none; position:fixed; z-index:10001;
              background:rgba(6,14,28,0.97);
              border-radius:16px; padding:24px;
              max-width:320px; width:min(320px,90vw);
              border:1px solid rgba(56,232,255,0.38);
              box-shadow:0 0 20px rgba(56,232,255,0.14),0 8px 32px rgba(0,0,0,0.78);
              font-family:'Inter',system-ui,sans-serif;
              backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <span id="tour-counter"
            style="font-size:11px; font-weight:700; letter-spacing:.08em;
                   color:rgba(56,232,255,0.65); text-transform:uppercase;"></span>
      <button onclick="window.tourFinish()"
              style="background:none; border:none; font-size:22px; cursor:pointer;
                     color:#506070; line-height:1; padding:0;"
              onmouseover="this.style.color='#38E8FF'"
              onmouseout="this.style.color='#506070'">&times;</button>
    </div>
    <h3 id="tour-title"
        style="margin:0 0 8px; font-size:17px; font-weight:700;
               color:#F0F4FF; letter-spacing:0.01em;"></h3>
    <p id="tour-text"
       style="margin:0 0 20px; font-size:14px; line-height:1.65; color:#8FA8C8;"></p>
    <div style="display:flex; gap:8px; justify-content:flex-end;">
      <button id="btn-back" onclick="window.tourBack()"
              style="padding:9px 18px; border-radius:9px;
                     border:1px solid rgba(56,232,255,0.35);
                     background:rgba(56,232,255,0.06); cursor:pointer;
                     font-size:13px; font-weight:600; color:#8FA8C8;"
              onmouseover="this.style.borderColor='rgba(56,232,255,0.7)';this.style.color='#38E8FF';"
              onmouseout="this.style.borderColor='rgba(56,232,255,0.35)';this.style.color='#8FA8C8';">
        Back
      </button>
      <button id="btn-next" onclick="window.tourNext()"
              style="padding:9px 20px; border-radius:9px; border:none;
                     background:linear-gradient(135deg,#00D9FF,#7C3AED);
                     color:#040E1C; cursor:pointer; font-size:13px; font-weight:700;
                     box-shadow:0 0 14px rgba(56,232,255,0.32);"
              onmouseover="this.style.filter='brightness(1.1)'"
              onmouseout="this.style.filter='none'">Next</button>
      <button id="btn-finish" onclick="window.tourFinish()"
              style="display:none; padding:9px 20px; border-radius:9px; border:none;
                     background:linear-gradient(135deg,#00D9FF,#7C3AED);
                     color:#040E1C; cursor:pointer; font-size:13px; font-weight:700;
                     box-shadow:0 0 14px rgba(56,232,255,0.32);"
              onmouseover="this.style.filter='brightness(1.1)'"
              onmouseout="this.style.filter='none'">Finish Tour</button>
    </div>
  </div>
  <button id="tour-reopen" onclick="window.tourStart()"
          style="display:none; position:fixed; bottom:24px; right:24px; z-index:9998;
                 background:linear-gradient(135deg,#00D9FF,#7C3AED);
                 color:#040E1C; border:none; border-radius:50px; padding:11px 22px;
                 font-size:13px; font-weight:700; cursor:pointer;
                 box-shadow:0 0 18px rgba(56,232,255,0.42),0 4px 14px rgba(0,0,0,0.4);
                 font-family:'Inter',system-ui,sans-serif;"
          onmouseover="this.style.filter='brightness(1.1)';this.style.transform='translateY(-2px)';"
          onmouseout="this.style.filter='none';this.style.transform='none';">
    ✦ Take the Tour
  </button>
</div>
"""

# ---------------------------------------------------------------------------
# Decorative background layers — all position:fixed, moved to <body> by JS
# ---------------------------------------------------------------------------
BG_DECOR_HTML = """
<div id="bg-decor" aria-hidden="true" style="
    position:fixed; bottom:0; right:0; width:750px; height:700px;
    pointer-events:none; z-index:0; overflow:hidden;">
  <svg width="750" height="700" viewBox="0 0 750 700"
       fill="none" xmlns="http://www.w3.org/2000/svg"
       style="position:absolute;right:0;bottom:0;">
    <style>
      @keyframes sp{0%{stroke-dashoffset:950;opacity:0}8%{opacity:.20}92%{opacity:.20}100%{stroke-dashoffset:0;opacity:0}}
      @keyframes dp{0%,100%{opacity:.45}50%{opacity:1}}
      .sp1{stroke-dasharray:950;animation:sp 9s .4s linear infinite}
      .sp2{stroke-dasharray:950;animation:sp 9s 3.4s linear infinite}
      .sp3{stroke-dasharray:950;animation:sp 9s 6.4s linear infinite}
      .dp1{animation:dp 2.8s 0.0s ease-in-out infinite}
      .dp2{animation:dp 2.8s 0.4s ease-in-out infinite}
      .dp3{animation:dp 2.8s 0.8s ease-in-out infinite}
      .dp4{animation:dp 2.8s 1.2s ease-in-out infinite}
      .dp5{animation:dp 2.8s 1.6s ease-in-out infinite}
      .dp6{animation:dp 2.8s 2.0s ease-in-out infinite}
      .dp7{animation:dp 2.8s 2.4s ease-in-out infinite}
    </style>
    <defs>
      <radialGradient id="bG1" cx="65%" cy="72%" r="50%">
        <stop offset="0%" stop-color="#38E8FF" stop-opacity="0.13"/>
        <stop offset="100%" stop-color="#38E8FF" stop-opacity="0"/>
      </radialGradient>
      <radialGradient id="bG2" cx="55%" cy="85%" r="50%">
        <stop offset="0%" stop-color="#7C3AED" stop-opacity="0.11"/>
        <stop offset="100%" stop-color="#7C3AED" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <ellipse cx="620" cy="620" rx="390" ry="290" stroke="rgba(56,232,255,0.055)" stroke-width="1" fill="none"/>
    <ellipse cx="620" cy="620" rx="290" ry="210" stroke="rgba(124,58,237,0.07)"  stroke-width="1" fill="none"/>
    <ellipse cx="620" cy="620" rx="195" ry="145" stroke="rgba(56,232,255,0.048)" stroke-width="1" fill="none"/>
    <ellipse cx="620" cy="620" rx="105" ry="80"  stroke="rgba(124,58,237,0.055)" stroke-width="1" fill="none"/>
    <path class="sp1" d="M 180 700 C 380 590 540 440 750 310" stroke="rgba(56,232,255,0.16)"  stroke-width="1.5" fill="none"/>
    <path class="sp2" d="M  80 700 C 340 540 520 370 750 210" stroke="rgba(124,58,237,0.14)" stroke-width="1.2" fill="none"/>
    <path class="sp3" d="M 290 700 C 470 610 610 510 750 415" stroke="rgba(56,232,255,0.12)"  stroke-width="1.2" fill="none"/>
    <path d="M 410 700 C 540 645 650 570 750 500" stroke="rgba(168,85,247,0.055)" stroke-width="0.8" fill="none"/>
    <path d="M 520 700 C 610 670 690 630 750 590" stroke="rgba(56,232,255,0.04)"  stroke-width="0.7" fill="none"/>
    <ellipse cx="670" cy="650" rx="270" ry="210" fill="url(#bG1)"/>
    <ellipse cx="530" cy="690" rx="210" ry="165" fill="url(#bG2)"/>
    <circle class="dp1" cx="500" cy="390" r="2.5" fill="rgba(56,232,255,0.52)"/>
    <circle class="dp2" cx="548" cy="432" r="1.8" fill="rgba(56,232,255,0.44)"/>
    <circle class="dp3" cx="592" cy="402" r="2.2" fill="rgba(56,232,255,0.38)"/>
    <circle class="dp4" cx="638" cy="366" r="1.8" fill="rgba(56,232,255,0.28)"/>
    <circle class="dp5" cx="564" cy="488" r="2.0" fill="rgba(124,58,237,0.52)"/>
    <circle class="dp6" cx="612" cy="508" r="2.5" fill="rgba(124,58,237,0.40)"/>
    <circle class="dp7" cx="662" cy="460" r="1.8" fill="rgba(124,58,237,0.28)"/>
    <line x1="476" y1="353" x2="486" y2="363" stroke="rgba(56,232,255,0.28)" stroke-width="1"/>
    <line x1="486" y1="353" x2="476" y2="363" stroke="rgba(56,232,255,0.28)" stroke-width="1"/>
    <line x1="682" y1="374" x2="690" y2="382" stroke="rgba(56,232,255,0.18)" stroke-width="1"/>
    <line x1="690" y1="374" x2="682" y2="382" stroke="rgba(56,232,255,0.18)" stroke-width="1"/>
  </svg>
</div>
"""

AMBIENT_HTML = """
<div id="ambient-layer" aria-hidden="true" style="
    position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
  <div style="position:absolute;top:7%;left:3%;width:420px;height:420px;border-radius:50%;
              background:radial-gradient(circle,rgba(56,232,255,0.072) 0%,transparent 70%);
              filter:blur(60px);animation:af1 30s ease-in-out infinite;"></div>
  <div style="position:absolute;bottom:8%;right:5%;width:500px;height:500px;border-radius:50%;
              background:radial-gradient(circle,rgba(124,58,237,0.068) 0%,transparent 70%);
              filter:blur(70px);animation:af2 36s ease-in-out infinite;"></div>
  <div style="position:absolute;top:42%;right:10%;width:280px;height:280px;border-radius:50%;
              background:radial-gradient(circle,rgba(56,232,255,0.052) 0%,transparent 70%);
              filter:blur(50px);animation:af3 24s ease-in-out infinite;animation-delay:-9s;"></div>
  <div style="position:absolute;top:4%;right:22%;width:340px;height:340px;border-radius:50%;
              background:radial-gradient(circle,rgba(167,139,250,0.058) 0%,transparent 70%);
              filter:blur(65px);animation:af4 40s ease-in-out infinite;animation-delay:-16s;"></div>
  <div style="position:absolute;bottom:22%;left:14%;width:260px;height:260px;border-radius:50%;
              background:radial-gradient(circle,rgba(56,232,255,0.042) 0%,transparent 70%);
              filter:blur(48px);animation:af5 32s ease-in-out infinite;animation-delay:-6s;"></div>
</div>
"""

SHIELD_ANIM_HTML = """
<div id="shield-anim" aria-hidden="true" style="
    position:fixed;top:14px;left:50%;
    transform:translateX(-50%);
    width:440px;height:440px;
    pointer-events:none;z-index:0;opacity:0.62;">
  <svg width="440" height="440" viewBox="-130 -130 260 260"
       xmlns="http://www.w3.org/2000/svg" overflow="visible">
    <style>
      #sg { animation:sgFloat 7s ease-in-out infinite; }
      #og { animation:ogSpin 10s linear infinite;
            transform-box:fill-box; transform-origin:50% 50%; }
      .pt { animation:ptWink 2.8s ease-in-out infinite; }
      .pr { animation:prPulse 6s ease-out infinite;
            transform-box:fill-box; transform-origin:50% 50%; }
      @keyframes sgFloat { 0%,100% { transform:translateY(0px); } 50% { transform:translateY(-9px); } }
      @keyframes ogSpin  { from { transform:skewX(-18deg) rotate(0deg); } to { transform:skewX(-18deg) rotate(360deg); } }
      @keyframes ptWink  { 0%,100% { opacity:0.28; } 50% { opacity:1.00; } }
      @keyframes prPulse { 0% { transform:scale(1.0); opacity:0.60; } 100% { transform:scale(2.6); opacity:0.00; } }
      @media (prefers-reduced-motion:reduce) { #sg,#og,.pt,.pr { animation:none !important; } }
    </style>
    <defs>
      <radialGradient id="saAura" cx="50%" cy="50%" r="55%">
        <stop offset="0%"   stop-color="#1040CC" stop-opacity="0.22"/>
        <stop offset="55%"  stop-color="#38E8FF" stop-opacity="0.07"/>
        <stop offset="100%" stop-color="#38E8FF" stop-opacity="0.00"/>
      </radialGradient>
      <linearGradient id="saBody" x1="25%" y1="0%" x2="75%" y2="100%">
        <stop offset="0%"   stop-color="#1B5EEA" stop-opacity="0.96"/>
        <stop offset="45%"  stop-color="#0C1D60" stop-opacity="0.98"/>
        <stop offset="100%" stop-color="#420E8A" stop-opacity="0.96"/>
      </linearGradient>
      <linearGradient id="saEdge" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%"   stop-color="#38E8FF"/>
        <stop offset="100%" stop-color="#8B5CF6"/>
      </linearGradient>
      <filter id="saG3" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <filter id="saG8" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="8" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <circle cx="0" cy="0" r="120" fill="url(#saAura)"/>
    <circle class="pr" style="animation-delay:0s"  cx="0" cy="0" r="50" fill="none" stroke="rgba(56,232,255,0.55)" stroke-width="1.6"/>
    <circle class="pr" style="animation-delay:2s"  cx="0" cy="0" r="50" fill="none" stroke="rgba(56,232,255,0.55)" stroke-width="1.6"/>
    <circle class="pr" style="animation-delay:4s"  cx="0" cy="0" r="50" fill="none" stroke="rgba(139,92,246,0.50)" stroke-width="1.4"/>
    <g id="og">
      <ellipse cx="0" cy="0" rx="82" ry="26" fill="none" stroke="rgba(56,232,255,0.10)" stroke-width="1.0"/>
      <ellipse cx="0" cy="0" rx="82" ry="26" fill="none" stroke="rgba(56,232,255,0.82)" stroke-width="2.2"
               stroke-dasharray="68 188" stroke-linecap="round" filter="url(#saG3)"/>
      <ellipse cx="0" cy="0" rx="82" ry="26" fill="none" stroke="rgba(139,92,246,0.72)" stroke-width="1.8"
               stroke-dasharray="44 212" stroke-dashoffset="-130" stroke-linecap="round" filter="url(#saG3)"/>
      <circle class="pt" style="animation-delay:0.0s;animation-duration:2.4s" cx=" 82" cy="  0" r="4.0" fill="#38E8FF" filter="url(#saG3)"/>
      <circle class="pt" style="animation-delay:0.5s;animation-duration:3.0s" cx=" 58" cy=" 18" r="2.5" fill="#38E8FF"/>
      <circle class="pt" style="animation-delay:1.0s;animation-duration:2.6s" cx="  0" cy=" 26" r="3.0" fill="#38E8FF" filter="url(#saG3)" opacity="0.85"/>
      <circle class="pt" style="animation-delay:1.5s;animation-duration:3.2s" cx="-58" cy=" 18" r="2.2" fill="#8B5CF6" opacity="0.80"/>
      <circle class="pt" style="animation-delay:0.3s;animation-duration:2.8s" cx="-82" cy="  0" r="3.5" fill="#8B5CF6" filter="url(#saG3)"/>
      <circle class="pt" style="animation-delay:0.8s;animation-duration:2.4s" cx="-58" cy="-18" r="2.0" fill="#8B5CF6" opacity="0.70"/>
      <circle class="pt" style="animation-delay:1.3s;animation-duration:3.0s" cx="  0" cy="-26" r="2.6" fill="#38E8FF" opacity="0.80"/>
      <circle class="pt" style="animation-delay:1.8s;animation-duration:2.6s" cx=" 58" cy="-18" r="2.0" fill="#38E8FF" opacity="0.55"/>
    </g>
    <g id="sg">
      <path d="M0-57 L46-41 L46-9 C46 24 25 39 0 47 C-25 39-46 24-46-9 L-46-41 Z"
            fill="rgba(56,232,255,0.04)" filter="url(#saG8)"/>
      <path d="M0-52 L41-37 L41-8 C41 21 23 34 0 42 C-23 34-41 21-41-8 L-41-37 Z"
            fill="url(#saBody)" stroke="url(#saEdge)" stroke-width="2.2"/>
      <path d="M0-44 L34-31 L34-5 C34 17 18 28 0 35 C-18 28-34 17-34-5 L-34-31 Z"
            fill="none" stroke="rgba(56,232,255,0.30)" stroke-width="0.9"/>
      <line x1="-22" y1="-42" x2="22" y2="-42" stroke="rgba(56,232,255,0.56)" stroke-width="0.9"/>
      <line x1="-39" y1="-32" x2="-33" y2="-36" stroke="rgba(56,232,255,0.42)" stroke-width="0.8"/>
      <line x1=" 33" y1="-36" x2=" 39" y2="-32" stroke="rgba(56,232,255,0.42)" stroke-width="0.8"/>
      <circle cx="0" cy="-3" r="21" fill="rgba(3,6,26,0.90)" stroke="rgba(56,232,255,0.68)" stroke-width="1.7" filter="url(#saG3)"/>
      <circle cx="0" cy="-3" r="14" fill="none" stroke="rgba(56,232,255,0.22)" stroke-width="0.8" stroke-dasharray="3.5 3.5"/>
      <circle cx="0" cy="-3" r="9.5" fill="rgba(100,28,158,0.60)" stroke="rgba(139,92,246,0.60)" stroke-width="1.1"/>
      <circle cx="0" cy="-3" r="5.2" fill="#38E8FF" opacity="0.92" filter="url(#saG3)"/>
      <circle cx="-5" cy="-8" r="2.0" fill="rgba(255,255,255,0.80)"/>
      <circle cx=" 3" cy=" 0" r="1.1" fill="rgba(255,255,255,0.40)"/>
      <circle cx="-7" cy="29" r="1.4" fill="rgba(56,232,255,0.55)"/>
      <circle cx=" 0" cy="31" r="1.4" fill="rgba(56,232,255,0.82)"/>
      <circle cx=" 7" cy="29" r="1.4" fill="rgba(56,232,255,0.55)"/>
    </g>
  </svg>
</div>
"""

BG_SHIELDS_HTML = """
<div id="bg-shields" aria-hidden="true" style="
    position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
  <div style="position:absolute;top:5%;left:0%;width:320px;height:381px;
              opacity:0.20;filter:blur(1px);animation:bsFloat1 32s ease-in-out infinite;">
    <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
            fill="rgba(7,18,38,0.40)" stroke="rgba(56,232,255,0.70)" stroke-width="1.8"/>
      <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
            fill="none" stroke="rgba(56,232,255,0.28)" stroke-width="0.8"/>
      <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(56,232,255,0.40)" stroke-width="1"/>
      <line x1="-49" y1="-37" x2="-40" y2="-42" stroke="rgba(56,232,255,0.45)" stroke-width="1.2"/>
      <line x1="40" y1="-42" x2="49" y2="-37" stroke="rgba(56,232,255,0.45)" stroke-width="1.2"/>
      <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(56,232,255,0.30)" stroke-width="0.8" stroke-dasharray="10 8"/>
      <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(56,232,255,0.55)" stroke-width="1.3"/>
      <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.30)" stroke="rgba(139,92,246,0.40)" stroke-width="0.9"/>
      <circle cx="0" cy="-5" r="6"  fill="rgba(56,232,255,0.22)" stroke="rgba(56,232,255,0.55)" stroke-width="0.7"/>
      <circle cx="-2" cy="-7" r="2.2" fill="rgba(255,255,255,0.75)"/>
      <circle cx="-8" cy="41" r="1.8" fill="rgba(56,232,255,0.70)"/>
      <circle cx="0"  cy="43" r="1.8" fill="rgba(56,232,255,0.88)"/>
      <circle cx="8"  cy="41" r="1.8" fill="rgba(56,232,255,0.70)"/>
    </svg>
  </div>
  <div style="position:absolute;top:2%;right:2%;width:285px;height:339px;
              opacity:0.16;filter:blur(2px);animation:bsFloat2 44s ease-in-out infinite;animation-delay:-10s;">
    <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
            fill="rgba(50,10,80,0.40)" stroke="rgba(139,92,246,0.72)" stroke-width="1.8"/>
      <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
            fill="none" stroke="rgba(139,92,246,0.28)" stroke-width="0.8"/>
      <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(139,92,246,0.40)" stroke-width="1"/>
      <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(139,92,246,0.30)" stroke-width="0.8" stroke-dasharray="10 8"/>
      <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(139,92,246,0.55)" stroke-width="1.3"/>
      <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.32)" stroke="rgba(56,232,255,0.40)" stroke-width="0.9"/>
      <circle cx="0" cy="-5" r="6"  fill="rgba(139,92,246,0.22)" stroke="rgba(139,92,246,0.55)" stroke-width="0.7"/>
      <circle cx="-8" cy="41" r="1.8" fill="rgba(139,92,246,0.70)"/>
      <circle cx="0"  cy="43" r="1.8" fill="rgba(139,92,246,0.88)"/>
      <circle cx="8"  cy="41" r="1.8" fill="rgba(139,92,246,0.70)"/>
    </svg>
  </div>
  <div style="position:absolute;top:42%;left:2%;width:220px;height:262px;
              opacity:0.16;filter:blur(1px);animation:bsFloat3 28s ease-in-out infinite;animation-delay:-5s;">
    <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
            fill="rgba(7,18,38,0.40)" stroke="rgba(56,232,255,0.65)" stroke-width="1.8"/>
      <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
            fill="none" stroke="rgba(56,232,255,0.25)" stroke-width="0.8"/>
      <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(56,232,255,0.28)" stroke-width="0.7" stroke-dasharray="8 7"/>
      <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(56,232,255,0.50)" stroke-width="1.2"/>
      <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.28)" stroke="rgba(139,92,246,0.38)" stroke-width="0.8"/>
      <circle cx="0" cy="-5" r="6"  fill="rgba(56,232,255,0.20)" stroke="rgba(56,232,255,0.50)" stroke-width="0.6"/>
      <circle cx="0" cy="43" r="1.8" fill="rgba(56,232,255,0.85)"/>
    </svg>
  </div>
  <div style="position:absolute;top:35%;right:1%;width:290px;height:345px;
              opacity:0.18;filter:blur(1.5px);animation:bsFloat4 38s ease-in-out infinite;animation-delay:-14s;">
    <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
            fill="rgba(50,10,80,0.35)" stroke="rgba(139,92,246,0.68)" stroke-width="1.8"/>
      <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
            fill="none" stroke="rgba(139,92,246,0.26)" stroke-width="0.8"/>
      <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(139,92,246,0.38)" stroke-width="0.9"/>
      <line x1="-49" y1="-37" x2="-40" y2="-42" stroke="rgba(139,92,246,0.42)" stroke-width="1"/>
      <line x1="40" y1="-42" x2="49" y2="-37" stroke="rgba(139,92,246,0.42)" stroke-width="1"/>
      <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(139,92,246,0.28)" stroke-width="0.7" stroke-dasharray="9 7"/>
      <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(139,92,246,0.50)" stroke-width="1.2"/>
      <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.30)" stroke="rgba(56,232,255,0.38)" stroke-width="0.8"/>
      <circle cx="0" cy="-5" r="6"  fill="rgba(139,92,246,0.20)" stroke="rgba(139,92,246,0.50)" stroke-width="0.6"/>
      <circle cx="-8" cy="41" r="1.8" fill="rgba(139,92,246,0.68)"/>
      <circle cx="0"  cy="43" r="1.8" fill="rgba(139,92,246,0.85)"/>
      <circle cx="8"  cy="41" r="1.8" fill="rgba(139,92,246,0.68)"/>
    </svg>
  </div>
  <div style="position:absolute;bottom:6%;left:6%;width:195px;height:232px;
              opacity:0.22;filter:blur(0.5px);animation:bsFloat5 22s ease-in-out infinite;animation-delay:-3s;">
    <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
            fill="rgba(7,18,38,0.42)" stroke="rgba(56,232,255,0.78)" stroke-width="2"/>
      <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
            fill="none" stroke="rgba(56,232,255,0.32)" stroke-width="0.9"/>
      <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(56,232,255,0.50)" stroke-width="1.1"/>
      <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.65)" stroke="rgba(56,232,255,0.62)" stroke-width="1.4"/>
      <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.35)" stroke="rgba(139,92,246,0.48)" stroke-width="1"/>
      <circle cx="0" cy="-5" r="6"  fill="rgba(56,232,255,0.25)" stroke="rgba(56,232,255,0.62)" stroke-width="0.8"/>
      <circle cx="-1.5" cy="-7" r="2.2" fill="rgba(255,255,255,0.80)"/>
      <circle cx="0"    cy="43" r="1.8" fill="rgba(56,232,255,0.90)"/>
    </svg>
  </div>
  <div style="position:absolute;top:18%;left:-95px;width:460px;height:265px;
              opacity:0.10;animation:bgOrbSpin1 22s linear infinite;">
    <svg viewBox="0 0 390 225" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <ellipse cx="195" cy="112" rx="188" ry="106" fill="none" stroke="rgba(56,232,255,0.90)" stroke-width="1.5" stroke-dasharray="30 22"/>
      <ellipse cx="195" cy="112" rx="145" ry="82"  fill="none" stroke="rgba(56,232,255,0.50)" stroke-width="0.8" stroke-dasharray="14 26"/>
      <circle cx="383" cy="112" r="4"   fill="rgba(56,232,255,0.90)"/>
      <circle cx="7"   cy="112" r="3"   fill="rgba(56,232,255,0.70)"/>
      <circle cx="195" cy="6"   r="3.5" fill="rgba(56,232,255,0.65)"/>
    </svg>
  </div>
  <div style="position:absolute;bottom:3%;left:25%;width:420px;height:235px;
              opacity:0.09;animation:bgOrbSpin2 28s linear infinite;animation-delay:-7s;">
    <svg viewBox="0 0 350 196" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
      <ellipse cx="175" cy="98" rx="168" ry="91" fill="none" stroke="rgba(139,92,246,0.90)" stroke-width="1.5" stroke-dasharray="26 18"/>
      <ellipse cx="175" cy="98" rx="125" ry="68" fill="none" stroke="rgba(139,92,246,0.50)" stroke-width="0.8" stroke-dasharray="12 22"/>
      <circle cx="343" cy="98" r="3.5" fill="rgba(139,92,246,0.90)"/>
      <circle cx="7"   cy="98" r="2.5" fill="rgba(139,92,246,0.70)"/>
    </svg>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# Welcome splash page — full-screen overlay shown on first load.
# All heavy animations live here so the home page stays lightweight.
# JS calls window.accessTool() to fade this out and start the tour.
# ---------------------------------------------------------------------------
WELCOME_PAGE_HTML = """
<div id="page-welcome" style="
    position:fixed;inset:0;z-index:500;
    background:#060E1C;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    overflow:hidden;
    transition:opacity 0.65s ease;">

  <!-- Circuit / arc lines ─── bottom-right -->
  <div id="bg-decor" aria-hidden="true" style="
      position:absolute;bottom:0;right:0;width:750px;height:700px;
      pointer-events:none;z-index:0;overflow:hidden;">
    <svg width="750" height="700" viewBox="0 0 750 700"
         fill="none" xmlns="http://www.w3.org/2000/svg"
         style="position:absolute;right:0;bottom:0;">
      <style>
        @keyframes sp{0%{stroke-dashoffset:950;opacity:0}8%{opacity:.20}92%{opacity:.20}100%{stroke-dashoffset:0;opacity:0}}
        @keyframes dp{0%,100%{opacity:.45}50%{opacity:1}}
        .sp1{stroke-dasharray:950;animation:sp 9s .4s linear infinite}
        .sp2{stroke-dasharray:950;animation:sp 9s 3.4s linear infinite}
        .sp3{stroke-dasharray:950;animation:sp 9s 6.4s linear infinite}
        .dp1{animation:dp 2.8s 0.0s ease-in-out infinite}
        .dp2{animation:dp 2.8s 0.4s ease-in-out infinite}
        .dp3{animation:dp 2.8s 0.8s ease-in-out infinite}
        .dp4{animation:dp 2.8s 1.2s ease-in-out infinite}
        .dp5{animation:dp 2.8s 1.6s ease-in-out infinite}
        .dp6{animation:dp 2.8s 2.0s ease-in-out infinite}
        .dp7{animation:dp 2.8s 2.4s ease-in-out infinite}
      </style>
      <defs>
        <radialGradient id="bG1" cx="65%" cy="72%" r="50%">
          <stop offset="0%" stop-color="#38E8FF" stop-opacity="0.13"/>
          <stop offset="100%" stop-color="#38E8FF" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="bG2" cx="55%" cy="85%" r="50%">
          <stop offset="0%" stop-color="#7C3AED" stop-opacity="0.11"/>
          <stop offset="100%" stop-color="#7C3AED" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <ellipse cx="620" cy="620" rx="390" ry="290" stroke="rgba(56,232,255,0.055)" stroke-width="1" fill="none"/>
      <ellipse cx="620" cy="620" rx="290" ry="210" stroke="rgba(124,58,237,0.07)"  stroke-width="1" fill="none"/>
      <ellipse cx="620" cy="620" rx="195" ry="145" stroke="rgba(56,232,255,0.048)" stroke-width="1" fill="none"/>
      <ellipse cx="620" cy="620" rx="105" ry="80"  stroke="rgba(124,58,237,0.055)" stroke-width="1" fill="none"/>
      <path class="sp1" d="M 180 700 C 380 590 540 440 750 310" stroke="rgba(56,232,255,0.16)"  stroke-width="1.5" fill="none"/>
      <path class="sp2" d="M  80 700 C 340 540 520 370 750 210" stroke="rgba(124,58,237,0.14)" stroke-width="1.2" fill="none"/>
      <path class="sp3" d="M 290 700 C 470 610 610 510 750 415" stroke="rgba(56,232,255,0.12)"  stroke-width="1.2" fill="none"/>
      <path d="M 410 700 C 540 645 650 570 750 500" stroke="rgba(168,85,247,0.055)" stroke-width="0.8" fill="none"/>
      <path d="M 520 700 C 610 670 690 630 750 590" stroke="rgba(56,232,255,0.04)"  stroke-width="0.7" fill="none"/>
      <ellipse cx="670" cy="650" rx="270" ry="210" fill="url(#bG1)"/>
      <ellipse cx="530" cy="690" rx="210" ry="165" fill="url(#bG2)"/>
      <circle class="dp1" cx="500" cy="390" r="2.5" fill="rgba(56,232,255,0.52)"/>
      <circle class="dp2" cx="548" cy="432" r="1.8" fill="rgba(56,232,255,0.44)"/>
      <circle class="dp3" cx="592" cy="402" r="2.2" fill="rgba(56,232,255,0.38)"/>
      <circle class="dp4" cx="638" cy="366" r="1.8" fill="rgba(56,232,255,0.28)"/>
      <circle class="dp5" cx="564" cy="488" r="2.0" fill="rgba(124,58,237,0.52)"/>
      <circle class="dp6" cx="612" cy="508" r="2.5" fill="rgba(124,58,237,0.40)"/>
      <circle class="dp7" cx="662" cy="460" r="1.8" fill="rgba(124,58,237,0.28)"/>
      <line x1="476" y1="353" x2="486" y2="363" stroke="rgba(56,232,255,0.28)" stroke-width="1"/>
      <line x1="486" y1="353" x2="476" y2="363" stroke="rgba(56,232,255,0.28)" stroke-width="1"/>
      <line x1="682" y1="374" x2="690" y2="382" stroke="rgba(56,232,255,0.18)" stroke-width="1"/>
      <line x1="690" y1="374" x2="682" y2="382" stroke="rgba(56,232,255,0.18)" stroke-width="1"/>
    </svg>
  </div>

  <!-- Aurora colour blobs -->
  <div id="ambient-layer" aria-hidden="true" style="
      position:absolute;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
    <div style="position:absolute;top:7%;left:3%;width:420px;height:420px;border-radius:50%;
                background:radial-gradient(circle,rgba(56,232,255,0.072) 0%,transparent 70%);
                filter:blur(60px);animation:af1 30s ease-in-out infinite;"></div>
    <div style="position:absolute;bottom:8%;right:5%;width:500px;height:500px;border-radius:50%;
                background:radial-gradient(circle,rgba(124,58,237,0.068) 0%,transparent 70%);
                filter:blur(70px);animation:af2 36s ease-in-out infinite;"></div>
    <div style="position:absolute;top:42%;right:10%;width:280px;height:280px;border-radius:50%;
                background:radial-gradient(circle,rgba(56,232,255,0.052) 0%,transparent 70%);
                filter:blur(50px);animation:af3 24s ease-in-out infinite;animation-delay:-9s;"></div>
    <div style="position:absolute;top:4%;right:22%;width:340px;height:340px;border-radius:50%;
                background:radial-gradient(circle,rgba(167,139,250,0.058) 0%,transparent 70%);
                filter:blur(65px);animation:af4 40s ease-in-out infinite;animation-delay:-16s;"></div>
    <div style="position:absolute;bottom:22%;left:14%;width:260px;height:260px;border-radius:50%;
                background:radial-gradient(circle,rgba(56,232,255,0.042) 0%,transparent 70%);
                filter:blur(48px);animation:af5 32s ease-in-out infinite;animation-delay:-6s;"></div>
  </div>

  <!-- Holographic floating shields -->
  <div id="bg-shields" aria-hidden="true" style="
      position:absolute;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
    <div style="position:absolute;top:5%;left:0%;width:320px;height:381px;
                opacity:0.20;filter:blur(1px);animation:bsFloat1 32s ease-in-out infinite;">
      <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
              fill="rgba(7,18,38,0.40)" stroke="rgba(56,232,255,0.70)" stroke-width="1.8"/>
        <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
              fill="none" stroke="rgba(56,232,255,0.28)" stroke-width="0.8"/>
        <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(56,232,255,0.40)" stroke-width="1"/>
        <line x1="-49" y1="-37" x2="-40" y2="-42" stroke="rgba(56,232,255,0.45)" stroke-width="1.2"/>
        <line x1="40" y1="-42" x2="49" y2="-37" stroke="rgba(56,232,255,0.45)" stroke-width="1.2"/>
        <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(56,232,255,0.30)" stroke-width="0.8" stroke-dasharray="10 8"/>
        <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(56,232,255,0.55)" stroke-width="1.3"/>
        <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.30)" stroke="rgba(139,92,246,0.40)" stroke-width="0.9"/>
        <circle cx="0" cy="-5" r="6"  fill="rgba(56,232,255,0.22)" stroke="rgba(56,232,255,0.55)" stroke-width="0.7"/>
        <circle cx="-2" cy="-7" r="2.2" fill="rgba(255,255,255,0.75)"/>
        <circle cx="-8" cy="41" r="1.8" fill="rgba(56,232,255,0.70)"/>
        <circle cx="0"  cy="43" r="1.8" fill="rgba(56,232,255,0.88)"/>
        <circle cx="8"  cy="41" r="1.8" fill="rgba(56,232,255,0.70)"/>
      </svg>
    </div>
    <div style="position:absolute;top:2%;right:2%;width:285px;height:339px;
                opacity:0.16;filter:blur(2px);animation:bsFloat2 44s ease-in-out infinite;animation-delay:-10s;">
      <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
              fill="rgba(50,10,80,0.40)" stroke="rgba(139,92,246,0.72)" stroke-width="1.8"/>
        <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
              fill="none" stroke="rgba(139,92,246,0.28)" stroke-width="0.8"/>
        <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(139,92,246,0.40)" stroke-width="1"/>
        <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(139,92,246,0.30)" stroke-width="0.8" stroke-dasharray="10 8"/>
        <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(139,92,246,0.55)" stroke-width="1.3"/>
        <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.32)" stroke="rgba(56,232,255,0.40)" stroke-width="0.9"/>
        <circle cx="0" cy="-5" r="6"  fill="rgba(139,92,246,0.22)" stroke="rgba(139,92,246,0.55)" stroke-width="0.7"/>
        <circle cx="-8" cy="41" r="1.8" fill="rgba(139,92,246,0.70)"/>
        <circle cx="0"  cy="43" r="1.8" fill="rgba(139,92,246,0.88)"/>
        <circle cx="8"  cy="41" r="1.8" fill="rgba(139,92,246,0.70)"/>
      </svg>
    </div>
    <div style="position:absolute;top:42%;left:2%;width:220px;height:262px;
                opacity:0.16;filter:blur(1px);animation:bsFloat3 28s ease-in-out infinite;animation-delay:-5s;">
      <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
              fill="rgba(7,18,38,0.40)" stroke="rgba(56,232,255,0.65)" stroke-width="1.8"/>
        <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
              fill="none" stroke="rgba(56,232,255,0.25)" stroke-width="0.8"/>
        <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(56,232,255,0.28)" stroke-width="0.7" stroke-dasharray="8 7"/>
        <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(56,232,255,0.50)" stroke-width="1.2"/>
        <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.28)" stroke="rgba(139,92,246,0.38)" stroke-width="0.8"/>
        <circle cx="0" cy="-5" r="6"  fill="rgba(56,232,255,0.20)" stroke="rgba(56,232,255,0.50)" stroke-width="0.6"/>
        <circle cx="0" cy="43" r="1.8" fill="rgba(56,232,255,0.85)"/>
      </svg>
    </div>
    <div style="position:absolute;top:35%;right:1%;width:290px;height:345px;
                opacity:0.18;filter:blur(1.5px);animation:bsFloat4 38s ease-in-out infinite;animation-delay:-14s;">
      <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
              fill="rgba(50,10,80,0.35)" stroke="rgba(139,92,246,0.68)" stroke-width="1.8"/>
        <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
              fill="none" stroke="rgba(139,92,246,0.26)" stroke-width="0.8"/>
        <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(139,92,246,0.38)" stroke-width="0.9"/>
        <line x1="-49" y1="-37" x2="-40" y2="-42" stroke="rgba(139,92,246,0.42)" stroke-width="1"/>
        <line x1="40" y1="-42" x2="49" y2="-37" stroke="rgba(139,92,246,0.42)" stroke-width="1"/>
        <ellipse cx="0" cy="-5" rx="32" ry="10" fill="none" stroke="rgba(139,92,246,0.28)" stroke-width="0.7" stroke-dasharray="9 7"/>
        <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.60)" stroke="rgba(139,92,246,0.50)" stroke-width="1.2"/>
        <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.30)" stroke="rgba(56,232,255,0.38)" stroke-width="0.8"/>
        <circle cx="0" cy="-5" r="6"  fill="rgba(139,92,246,0.20)" stroke="rgba(139,92,246,0.50)" stroke-width="0.6"/>
        <circle cx="-8" cy="41" r="1.8" fill="rgba(139,92,246,0.68)"/>
        <circle cx="0"  cy="43" r="1.8" fill="rgba(139,92,246,0.85)"/>
        <circle cx="8"  cy="41" r="1.8" fill="rgba(139,92,246,0.68)"/>
      </svg>
    </div>
    <div style="position:absolute;bottom:6%;left:6%;width:195px;height:232px;
                opacity:0.22;filter:blur(0.5px);animation:bsFloat5 22s ease-in-out infinite;animation-delay:-3s;">
      <svg viewBox="-55 -65 110 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <path d="M0-60 L50-43 L50-12 C50 27 28 44 0 54 C-28 44-50 27-50-12 L-50-43 Z"
              fill="rgba(7,18,38,0.42)" stroke="rgba(56,232,255,0.78)" stroke-width="2"/>
        <path d="M0-50 L40-36 L40-9 C40 21 22 35 0 43 C-22 35-40 21-40-9 L-40-36 Z"
              fill="none" stroke="rgba(56,232,255,0.32)" stroke-width="0.9"/>
        <line x1="-24" y1="-57" x2="24" y2="-57" stroke="rgba(56,232,255,0.50)" stroke-width="1.1"/>
        <circle cx="0" cy="-5" r="21" fill="rgba(3,6,26,0.65)" stroke="rgba(56,232,255,0.62)" stroke-width="1.4"/>
        <circle cx="0" cy="-5" r="13" fill="rgba(100,28,158,0.35)" stroke="rgba(139,92,246,0.48)" stroke-width="1"/>
        <circle cx="0" cy="-5" r="6"  fill="rgba(56,232,255,0.25)" stroke="rgba(56,232,255,0.62)" stroke-width="0.8"/>
        <circle cx="-1.5" cy="-7" r="2.2" fill="rgba(255,255,255,0.80)"/>
        <circle cx="0"    cy="43" r="1.8" fill="rgba(56,232,255,0.90)"/>
      </svg>
    </div>
    <div style="position:absolute;top:18%;left:-95px;width:460px;height:265px;
                opacity:0.10;animation:bgOrbSpin1 22s linear infinite;">
      <svg viewBox="0 0 390 225" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <ellipse cx="195" cy="112" rx="188" ry="106" fill="none" stroke="rgba(56,232,255,0.90)" stroke-width="1.5" stroke-dasharray="30 22"/>
        <ellipse cx="195" cy="112" rx="145" ry="82"  fill="none" stroke="rgba(56,232,255,0.50)" stroke-width="0.8" stroke-dasharray="14 26"/>
        <circle cx="383" cy="112" r="4"   fill="rgba(56,232,255,0.90)"/>
        <circle cx="7"   cy="112" r="3"   fill="rgba(56,232,255,0.70)"/>
        <circle cx="195" cy="6"   r="3.5" fill="rgba(56,232,255,0.65)"/>
      </svg>
    </div>
    <div style="position:absolute;bottom:3%;left:25%;width:420px;height:235px;
                opacity:0.09;animation:bgOrbSpin2 28s linear infinite;animation-delay:-7s;">
      <svg viewBox="0 0 350 196" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <ellipse cx="175" cy="98" rx="168" ry="91" fill="none" stroke="rgba(139,92,246,0.90)" stroke-width="1.5" stroke-dasharray="26 18"/>
        <ellipse cx="175" cy="98" rx="125" ry="68" fill="none" stroke="rgba(139,92,246,0.50)" stroke-width="0.8" stroke-dasharray="12 22"/>
        <circle cx="343" cy="98" r="3.5" fill="rgba(139,92,246,0.90)"/>
        <circle cx="7"   cy="98" r="2.5" fill="rgba(139,92,246,0.70)"/>
      </svg>
    </div>
  </div>

  <!-- Central animated shield ─── in flex flow, above text -->
  <div id="shield-anim" aria-hidden="true" style="
      width:300px;height:300px;pointer-events:none;z-index:2;opacity:0.90;flex-shrink:0;">
    <svg width="100%" height="100%" viewBox="-130 -130 260 260"
         xmlns="http://www.w3.org/2000/svg" overflow="visible">
      <style>
        #sg { animation:sgFloat 7s ease-in-out infinite; }
        #og { animation:ogSpin 10s linear infinite;
              transform-box:fill-box; transform-origin:50% 50%; }
        .pt { animation:ptWink 2.8s ease-in-out infinite; }
        .pr { animation:prPulse 6s ease-out infinite;
              transform-box:fill-box; transform-origin:50% 50%; }
        @keyframes sgFloat { 0%,100% { transform:translateY(0px); } 50% { transform:translateY(-9px); } }
        @keyframes ogSpin  { from { transform:skewX(-18deg) rotate(0deg); } to { transform:skewX(-18deg) rotate(360deg); } }
        @keyframes ptWink  { 0%,100% { opacity:0.28; } 50% { opacity:1.00; } }
        @keyframes prPulse { 0% { transform:scale(1.0); opacity:0.60; } 100% { transform:scale(2.6); opacity:0.00; } }
        @media (prefers-reduced-motion:reduce) { #sg,#og,.pt,.pr { animation:none !important; } }
      </style>
      <defs>
        <radialGradient id="saAura" cx="50%" cy="50%" r="55%">
          <stop offset="0%"   stop-color="#1040CC" stop-opacity="0.22"/>
          <stop offset="55%"  stop-color="#38E8FF" stop-opacity="0.07"/>
          <stop offset="100%" stop-color="#38E8FF" stop-opacity="0.00"/>
        </radialGradient>
        <linearGradient id="saBody" x1="25%" y1="0%" x2="75%" y2="100%">
          <stop offset="0%"   stop-color="#1B5EEA" stop-opacity="0.96"/>
          <stop offset="45%"  stop-color="#0C1D60" stop-opacity="0.98"/>
          <stop offset="100%" stop-color="#420E8A" stop-opacity="0.96"/>
        </linearGradient>
        <linearGradient id="saEdge" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stop-color="#38E8FF"/>
          <stop offset="100%" stop-color="#8B5CF6"/>
        </linearGradient>
        <filter id="saG3" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="saG8" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="8" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <circle cx="0" cy="0" r="120" fill="url(#saAura)"/>
      <circle class="pr" style="animation-delay:0s"  cx="0" cy="0" r="50" fill="none" stroke="rgba(56,232,255,0.55)" stroke-width="1.6"/>
      <circle class="pr" style="animation-delay:2s"  cx="0" cy="0" r="50" fill="none" stroke="rgba(56,232,255,0.55)" stroke-width="1.6"/>
      <circle class="pr" style="animation-delay:4s"  cx="0" cy="0" r="50" fill="none" stroke="rgba(139,92,246,0.50)" stroke-width="1.4"/>
      <g id="og">
        <ellipse cx="0" cy="0" rx="82" ry="26" fill="none" stroke="rgba(56,232,255,0.10)" stroke-width="1.0"/>
        <ellipse cx="0" cy="0" rx="82" ry="26" fill="none" stroke="rgba(56,232,255,0.82)" stroke-width="2.2"
                 stroke-dasharray="68 188" stroke-linecap="round" filter="url(#saG3)"/>
        <ellipse cx="0" cy="0" rx="82" ry="26" fill="none" stroke="rgba(139,92,246,0.72)" stroke-width="1.8"
                 stroke-dasharray="44 212" stroke-dashoffset="-130" stroke-linecap="round" filter="url(#saG3)"/>
        <circle class="pt" style="animation-delay:0.0s;animation-duration:2.4s" cx=" 82" cy="  0" r="4.0" fill="#38E8FF" filter="url(#saG3)"/>
        <circle class="pt" style="animation-delay:0.5s;animation-duration:3.0s" cx=" 58" cy=" 18" r="2.5" fill="#38E8FF"/>
        <circle class="pt" style="animation-delay:1.0s;animation-duration:2.6s" cx="  0" cy=" 26" r="3.0" fill="#38E8FF" filter="url(#saG3)" opacity="0.85"/>
        <circle class="pt" style="animation-delay:1.5s;animation-duration:3.2s" cx="-58" cy=" 18" r="2.2" fill="#8B5CF6" opacity="0.80"/>
        <circle class="pt" style="animation-delay:0.3s;animation-duration:2.8s" cx="-82" cy="  0" r="3.5" fill="#8B5CF6" filter="url(#saG3)"/>
        <circle class="pt" style="animation-delay:0.8s;animation-duration:2.4s" cx="-58" cy="-18" r="2.0" fill="#8B5CF6" opacity="0.70"/>
        <circle class="pt" style="animation-delay:1.3s;animation-duration:3.0s" cx="  0" cy="-26" r="2.6" fill="#38E8FF" opacity="0.80"/>
        <circle class="pt" style="animation-delay:1.8s;animation-duration:2.6s" cx=" 58" cy="-18" r="2.0" fill="#38E8FF" opacity="0.55"/>
      </g>
      <g id="sg">
        <path d="M0-57 L46-41 L46-9 C46 24 25 39 0 47 C-25 39-46 24-46-9 L-46-41 Z"
              fill="rgba(56,232,255,0.04)" filter="url(#saG8)"/>
        <path d="M0-52 L41-37 L41-8 C41 21 23 34 0 42 C-23 34-41 21-41-8 L-41-37 Z"
              fill="url(#saBody)" stroke="url(#saEdge)" stroke-width="2.2"/>
        <path d="M0-44 L34-31 L34-5 C34 17 18 28 0 35 C-18 28-34 17-34-5 L-34-31 Z"
              fill="none" stroke="rgba(56,232,255,0.30)" stroke-width="0.9"/>
        <line x1="-22" y1="-42" x2="22" y2="-42" stroke="rgba(56,232,255,0.56)" stroke-width="0.9"/>
        <line x1="-39" y1="-32" x2="-33" y2="-36" stroke="rgba(56,232,255,0.42)" stroke-width="0.8"/>
        <line x1=" 33" y1="-36" x2=" 39" y2="-32" stroke="rgba(56,232,255,0.42)" stroke-width="0.8"/>
        <circle cx="0" cy="-3" r="21" fill="rgba(3,6,26,0.90)" stroke="rgba(56,232,255,0.68)" stroke-width="1.7" filter="url(#saG3)"/>
        <circle cx="0" cy="-3" r="14" fill="none" stroke="rgba(56,232,255,0.22)" stroke-width="0.8" stroke-dasharray="3.5 3.5"/>
        <circle cx="0" cy="-3" r="9.5" fill="rgba(100,28,158,0.60)" stroke="rgba(139,92,246,0.60)" stroke-width="1.1"/>
        <circle cx="0" cy="-3" r="5.2" fill="#38E8FF" opacity="0.92" filter="url(#saG3)"/>
        <circle cx="-5" cy="-8" r="2.0" fill="rgba(255,255,255,0.80)"/>
        <circle cx=" 3" cy=" 0" r="1.1" fill="rgba(255,255,255,0.40)"/>
        <circle cx="-7" cy="29" r="1.4" fill="rgba(56,232,255,0.55)"/>
        <circle cx=" 0" cy="31" r="1.4" fill="rgba(56,232,255,0.82)"/>
        <circle cx=" 7" cy="29" r="1.4" fill="rgba(56,232,255,0.55)"/>
      </g>
    </svg>
  </div>

  <!-- Welcome text ─── in flex flow, below shield -->
  <div id="welcome-content" style="
      position:relative;z-index:2;text-align:center;
      padding:0 40px 60px;max-width:520px;">
    <div style="font-size:0.72rem;letter-spacing:0.22em;font-weight:700;
                color:rgba(56,232,255,0.65);text-transform:uppercase;margin-bottom:8px;
                font-family:Inter,system-ui,sans-serif;">
      AI-Powered Image Protection
    </div>
    <h1 style="font-size:clamp(1.9rem,4.5vw,3rem);font-weight:800;margin:0 0 12px;
               background:linear-gradient(90deg,#38E8FF 0%,#A78BFA 50%,#6EE7FF 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;
               font-family:Inter,system-ui,sans-serif;letter-spacing:-0.01em;line-height:1.1;">
      Welcome to SafeShot
    </h1>
    <p style="color:#6E8CAA;font-size:0.92rem;margin:0 0 28px;line-height:1.6;
              font-family:Inter,system-ui,sans-serif;">
      Protect your images against AI editing models &mdash; with a single click.
    </p>
    <button id="access-tool-btn"
            onclick="window.accessTool()"
            onmouseover="this.style.filter='brightness(1.12)';this.style.transform='translateY(-2px)';"
            onmouseout="this.style.filter='none';this.style.transform='none';"
            style="
              background:linear-gradient(135deg,#00D9FF 0%,#38E8FF 45%,#7C3AED 100%);
              color:#040E1C;border:none;border-radius:50px;
              padding:14px 42px;font-size:1rem;font-weight:700;cursor:pointer;
              box-shadow:0 0 22px rgba(56,232,255,0.45),0 4px 18px rgba(0,0,0,0.4);
              font-family:Inter,system-ui,sans-serif;letter-spacing:0.04em;
              transition:filter 0.15s,transform 0.15s;">
      Access Tool &nbsp;&#8250;
    </button>
  </div>

</div>
"""

# ---------------------------------------------------------------------------
# All JavaScript — runs via demo.load() after Gradio finishes rendering
# ---------------------------------------------------------------------------
ALL_JS = """
() => {
    // ── Helper ───────────────────────────────────────────────────────────────
    function gid(id) { return document.getElementById(id); }

    // ── LAYER 3: Runtime CSS injection ───────────────────────────────────────
    var runtimeCSS = `
        .block .label-wrap, .form .label-wrap,
        .block > div > .label-wrap, .wrap > .label-wrap {
            display: inline-flex !important;
            align-items: center !important;
            background: rgba(56,232,255,0.07) !important;
            border: 1px solid rgba(56,232,255,0.42) !important;
            border-radius: 20px !important;
            padding: 3px 12px !important;
            margin-bottom: 10px !important;
            width: auto !important;
            max-width: max-content !important;
        }
        .block .label-wrap span, .block .label-wrap label span,
        .form .label-wrap span, .form .label-wrap label span {
            color: #38E8FF !important;
            -webkit-text-fill-color: #38E8FF !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
            background: none !important;
            -webkit-background-clip: unset !important;
        }
        #hero-header, div#hero-header, #hero-header .block {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }
        button.primary, button.primary:not(:disabled), .primary > button {
            background: linear-gradient(135deg,#00D9FF 0%,#38E8FF 45%,#7C3AED 100%) !important;
            color: #040E1C !important; border: none !important;
            font-weight: 700 !important; min-height: 52px !important;
        }
        button.stop, button.cancel, button.stop:not(:disabled),
        .stop > button, .cancel > button {
            background: linear-gradient(135deg,#FF8C42 0%,#FF5E5B 100%) !important;
            color: #ffffff !important; border: none !important;
            font-weight: 700 !important; min-height: 52px !important;
        }
        #tour-root, #page-welcome {
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            border: none !important; box-shadow: none !important;
            padding: 0 !important; margin: 0 !important;
        }
    `;
    var styleEl = document.createElement('style');
    styleEl.id = 'safeshot-runtime';
    styleEl.textContent = runtimeCSS;
    document.head.appendChild(styleEl);

    // ── Escape backdrop-filter stacking context ───────────────────────────────
    // Chrome/Safari create a new containing block for position:fixed children
    // when any ancestor has backdrop-filter. We re-attach decorative layers
    // directly to <body> so their fixed positioning works correctly.
    function escapeToBody(elId, insertBefore) {
        var el = document.getElementById(elId);
        if (!el) return;
        var ancestor = el.parentElement, blockWrapper = null;
        while (ancestor && ancestor !== document.body) {
            ancestor.style.setProperty('backdrop-filter',         'none', 'important');
            ancestor.style.setProperty('-webkit-backdrop-filter', 'none', 'important');
            ancestor.style.setProperty('background',              'transparent', 'important');
            ancestor.style.setProperty('border',                  'none', 'important');
            ancestor.style.setProperty('box-shadow',              'none', 'important');
            ancestor.style.setProperty('animation',               'none', 'important');
            ancestor.style.setProperty('padding',                 '0',    'important');
            ancestor.style.setProperty('margin',                  '0',    'important');
            var isBlock = ancestor.classList.contains('block') || ancestor.classList.contains('form');
            if (isBlock) blockWrapper = ancestor;
            ancestor = ancestor.parentElement;
            if (isBlock) break;
        }
        if (insertBefore) { document.body.insertBefore(el, document.body.firstChild); }
        else              { document.body.appendChild(el); }
        if (blockWrapper) {
            blockWrapper.style.setProperty('height',     '0',       'important');
            blockWrapper.style.setProperty('min-height', '0',       'important');
            blockWrapper.style.setProperty('overflow',   'hidden',  'important');
            blockWrapper.style.setProperty('display',    'block',   'important');
        }
    }

    // Move tour overlay and welcome page directly to body so
    // position:fixed children work correctly in Chrome / Safari
    escapeToBody('tour-root',    false);
    escapeToBody('page-welcome', false);

    // ── Dynamic element styles ────────────────────────────────────────────────
    function applyDynamicStyles() {
        // Hero header: transparent panel + gradient shimmer h1 + SVG shields
        var heroEl = document.getElementById('hero-header');
        if (heroEl) {
            var heroTargets = [heroEl];
            var inner = heroEl.querySelector('.block') || heroEl.querySelector('.wrap');
            if (inner) heroTargets.push(inner);
            heroTargets.forEach(function(el) {
                el.style.setProperty('background',              'transparent', 'important');
                el.style.setProperty('border',                  'none',        'important');
                el.style.setProperty('box-shadow',              'none',        'important');
                el.style.setProperty('backdrop-filter',         'none',        'important');
                el.style.setProperty('-webkit-backdrop-filter', 'none',        'important');
            });
            var h1 = heroEl.querySelector('h1');
            if (h1 && !h1.dataset.styled) {
                h1.dataset.styled = '1';
                var shSVG =
                    '<svg width="26" height="30" viewBox="-20 -23 40 47" xmlns="http://www.w3.org/2000/svg"' +
                    ' style="display:block;flex-shrink:0;filter:drop-shadow(0 0 5px rgba(56,232,255,0.70));">' +
                    '<path d="M0-22 L17-15 L17-4 C17 9 9 14 0 17 C-9 14-17 9-17-4 L-17-15 Z"' +
                    ' fill="rgba(10,20,55,0.95)" stroke="rgba(56,232,255,0.90)" stroke-width="1.4"/>' +
                    '<path d="M0-18 L13-12 L13-2 C13 6 7 10 0 13 C-7 10-13 6-13-2 L-13-12 Z"' +
                    ' fill="none" stroke="rgba(56,232,255,0.28)" stroke-width="0.7"/>' +
                    '<ellipse cx="0" cy="-2" rx="11" ry="3.5" fill="none"' +
                    ' stroke="rgba(56,232,255,0.28)" stroke-width="0.6" stroke-dasharray="4 4"/>' +
                    '<circle cx="0" cy="-2" r="8" fill="rgba(3,6,26,0.90)"' +
                    ' stroke="rgba(56,232,255,0.75)" stroke-width="1.1"/>' +
                    '<circle cx="0" cy="-2" r="5" fill="rgba(100,28,158,0.65)"' +
                    ' stroke="rgba(139,92,246,0.60)" stroke-width="0.8"/>' +
                    '<circle cx="0" cy="-2" r="2.5" fill="#38E8FF" opacity="0.95"/>' +
                    '<circle cx="-1.2" cy="-3.2" r="0.9" fill="rgba(255,255,255,0.88)"/>' +
                    '</svg>';
                var origText = h1.textContent.trim();
                var txtSpan =
                    '<span style="background:linear-gradient(90deg,#38E8FF 0%,#A78BFA 40%,#6EE7FF 65%,#A78BFA 100%);' +
                    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;' +
                    'background-clip:text;' +
                    'font-size:1.65rem;line-height:1.35;font-weight:700;' +
                    'font-family:Inter,system-ui,sans-serif;letter-spacing:0.02em;">' +
                    origText + '</span>';
                h1.style.cssText =
                    'display:flex;align-items:center;gap:10px;margin-bottom:0.4rem;' +
                    'filter:drop-shadow(0 0 18px rgba(56,232,255,0.28));flex-wrap:wrap;';
                h1.innerHTML = shSVG + txtSpan + shSVG;
            }
            var pEl = heroEl.querySelector('p');
            if (pEl) {
                pEl.style.setProperty('color',     '#6E8CAA', 'important');
                pEl.style.setProperty('font-size', '0.88rem', 'important');
            }
        }

        // Label pills
        document.querySelectorAll('.block .label-wrap, .form .label-wrap').forEach(function(el) {
            if (el.dataset.pillStyled) return;
            el.dataset.pillStyled = '1';
            el.style.setProperty('display',       'inline-flex',                  'important');
            el.style.setProperty('align-items',   'center',                       'important');
            el.style.setProperty('background',    'rgba(56,232,255,0.07)',        'important');
            el.style.setProperty('border',        '1px solid rgba(56,232,255,0.42)', 'important');
            el.style.setProperty('border-radius', '20px',                          'important');
            el.style.setProperty('padding',       '3px 12px',                     'important');
            el.style.setProperty('margin-bottom', '10px',                          'important');
            el.style.setProperty('width',         'auto',                          'important');
            el.style.setProperty('max-width',     'max-content',                  'important');
            el.querySelectorAll('span, label, label span').forEach(function(t) {
                t.style.setProperty('color',                   '#38E8FF',   'important');
                t.style.setProperty('-webkit-text-fill-color', '#38E8FF',   'important');
                t.style.setProperty('font-size',               '0.72rem',   'important');
                t.style.setProperty('font-weight',             '700',       'important');
                t.style.setProperty('letter-spacing',          '0.09em',    'important');
                t.style.setProperty('text-transform',          'uppercase', 'important');
                t.style.setProperty('background',              'none',      'important');
                t.style.setProperty('-webkit-background-clip', 'unset',     'important');
            });
        });
    }

    applyDynamicStyles();
    // Debounced: run at most once per 900ms so progress-bar DOM churn
    // during model runs doesn't hammer querySelectorAll on every mutation.
    var dstTimer;
    var domObserver = new MutationObserver(function() {
        clearTimeout(dstTimer);
        dstTimer = setTimeout(applyDynamicStyles, 900);
    });
    domObserver.observe(document.body, { childList: true, subtree: true });

    // ── Welcome page → tool transition ───────────────────────────────────────
    window.accessTool = function() {
        var welcome = gid('page-welcome');
        if (!welcome) { window.tourStart(); return; }
        welcome.style.opacity = '0';
        setTimeout(function() {
            welcome.style.display = 'none';
            window.tourStart();
        }, 680);
    };

    // ── Page navigation ───────────────────────────────────────────────────────
    window.showPage = function(page) {
        ['page-main','page-guide','page-about'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        ['nav-home','nav-guide','nav-about'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) { el.style.borderBottomColor = 'transparent'; el.style.color = '#C4D4E4'; el.style.textShadow = 'none'; }
        });

        var pageMap = { main: 'page-main', guide: 'page-guide', about: 'page-about' };
        var navMap  = { main: 'nav-home',  guide: 'nav-guide',  about: 'nav-about'  };
        var pageEl  = document.getElementById(pageMap[page]);
        var navEl   = document.getElementById(navMap[page]);
        if (pageEl) pageEl.style.display = (page === 'main') ? 'block' : 'flex';
        if (navEl)  { navEl.style.borderBottomColor = '#38E8FF'; navEl.style.color = '#38E8FF'; navEl.style.textShadow = '0 0 10px rgba(56,232,255,0.55)'; }

        var tourBtn = gid('tour-reopen');
        if (page === 'main') {
            if (tourBtn && tourFinished) tourBtn.style.display = 'block';
        } else {
            window.tourFinish && window.tourFinish(false);
            if (tourBtn) tourBtn.style.display = 'none';
        }
    };

    // Highlight Home tab on initial load
    var navH = document.getElementById('nav-home');
    if (navH) { navH.style.borderBottomColor = '#38E8FF'; navH.style.color = '#38E8FF'; navH.style.textShadow = '0 0 10px rgba(56,232,255,0.55)'; }

    // ── Spotlight tour ────────────────────────────────────────────────────────
    const STEPS = [
        {
            title: "Welcome to the SafeShot Tool!",
            text:  "This tool protects your image against AI image-editing models. " +
                   "Click Next to visit every feature.",
            targetId: null, pad: 0
        },
        {
            title: "Step 1 — Upload Your Image",
            text:  "Click this box or drag and drop to upload an image. " +
                   "Supported formats: JPG, PNG, WEBP.",
            targetId: "input_image", pad: 10
        },
        {
            title: "Step 2 — Choose Protection Mode",
            text:  "IP2P mode (default) targets InstructPix2Pix-based editors (EditShield). " +
                   "SD mode targets Stable Diffusion inpainting pipelines (BlurGuard).",
            targetId: "mode_selector", pad: 10
        },
        {
            title: "Step 3 — Output Resolution",
            text:  "Choose the image size fed into the protection model. " +
                   "'Original' keeps your source dimensions (rounded to nearest multiple of 8). " +
                   "Lower fixed sizes run faster and use less memory.",
            targetId: "resolution_selector", pad: 10
        },
        {
            title: "Step 4 — Perturbation Strength (ε)",
            text:  "Epsilon is the maximum pixel-level change allowed per channel. " +
                   "Higher values give stronger protection at the cost of slightly more visible noise. " +
                   "Resets to the recommended default when you switch modes.",
            targetId: "eps_slider", pad: 10
        },
        {
            title: "Step 5 — Optimization Steps",
            text:  "More steps refine the protection layer further but take longer. " +
                   "IP2P works well at 50 steps. For SD mode the warmup is ~1/3 of your " +
                   "step count, so the remaining 2/3 run the actual PGD attack.",
            targetId: "steps_slider", pad: 10
        },
        {
            title: "Step 6 — Protect Your Image",
            text:  "Once your image is loaded, click Protect Image to start. " +
                   "Use Stop to cancel between optimization steps.",
            targetId: "protect_btn", pad: 8
        },
        {
            title: "Step 7 — View Your Result",
            text:  "Your protected image appears here once the process completes.",
            targetId: "output_image", pad: 10
        },
        {
            title: "Step 8 — Download Your Protected Image",
            text:  "Click the download button below to save the protected image as a PNG.",
            targetId: "download_section", pad: 10
        }
    ];

    let currentStep = 0, tourFinished = false;

    function positionCard(targetId, pad) {
        const card = gid("tour-card"), svg = gid("tour-svg"), hole = gid("tour-hole");
        const vw = window.innerWidth, vh = window.innerHeight, MARGIN = 18;

        if (!targetId) {
            svg.style.display = "none";
            hole.setAttribute("width", 0); hole.setAttribute("height", 0);
            card.style.top = "50%"; card.style.left = "50%";
            card.style.transform = "translate(-50%,-50%)";
            return;
        }
        const target = document.getElementById(targetId);
        if (!target) return;

        var absTop = target.getBoundingClientRect().top + window.pageYOffset;
        window.scrollTo(0, Math.max(0, absTop - vh / 2 + target.offsetHeight / 2));

        card.style.transform = "";
        const r  = target.getBoundingClientRect();
        const cw = card.offsetWidth  > 0 ? card.offsetWidth  : Math.min(320, vw * 0.9);
        const ch = card.offsetHeight > 0 ? card.offsetHeight + 16 : 420;

        hole.setAttribute("x",      r.left   - pad);
        hole.setAttribute("y",      r.top    - pad);
        hole.setAttribute("width",  r.width  + pad * 2);
        hole.setAttribute("height", r.height + pad * 2);
        svg.style.display = "block";

        const fits = {
            below: r.bottom + MARGIN + ch <= vh,
            above: r.top    - MARGIN - ch >= 0,
            right: r.right  + MARGIN + cw <= vw,
            left:  r.left   - MARGIN - cw >= 0
        };
        let top, left;
        if      (fits.below) { top = r.bottom + MARGIN; left = Math.max(MARGIN, Math.min(r.left, vw - cw - MARGIN)); }
        else if (fits.above) { top = r.top - MARGIN - ch; left = Math.max(MARGIN, Math.min(r.left, vw - cw - MARGIN)); }
        else if (fits.right) { left = r.right + MARGIN; top = Math.max(MARGIN, Math.min(r.top + r.height / 2 - ch / 2, vh - ch - MARGIN)); }
        else if (fits.left)  { left = r.left - MARGIN - cw; top = Math.max(MARGIN, Math.min(r.top + r.height / 2 - ch / 2, vh - ch - MARGIN)); }
        else                 { top = r.bottom + MARGIN; left = Math.max(MARGIN, Math.min(r.left, vw - cw - MARGIN)); }

        card.style.top = top + "px"; card.style.left = left + "px";
    }

    function renderStep() {
        const s = STEPS[currentStep], total = STEPS.length;
        gid("tour-counter").textContent = (currentStep + 1) + " of " + total;
        gid("tour-title").textContent   = s.title;
        gid("tour-text").textContent    = s.text;
        gid("btn-back").style.display   = currentStep === 0           ? "none" : "";
        gid("btn-next").style.display   = currentStep === total - 1   ? "none" : "";
        gid("btn-finish").style.display = currentStep === total - 1   ? ""     : "none";
        setTimeout(() => positionCard(s.targetId, s.pad), 60);
    }

    window.tourStart = function() {
        currentStep = 0; tourFinished = false;
        gid("tour-blocker").style.display = "block";
        gid("tour-card").style.display    = "block";
        gid("tour-reopen").style.display  = "none";
        renderStep();
    };
    window.tourNext   = function() { if (currentStep < STEPS.length - 1) { currentStep++; renderStep(); } };
    window.tourBack   = function() { if (currentStep > 0) { currentStep--; renderStep(); } };
    window.tourFinish = function(showButton) {
        const show = showButton !== false;
        gid("tour-svg").style.display     = "none";
        gid("tour-blocker").style.display = "none";
        gid("tour-card").style.display    = "none";
        if (show) { tourFinished = true; gid("tour-reopen").style.display = "block"; }
    };

    window.addEventListener("scroll", () => {
        const s = STEPS[currentStep];
        if (gid("tour-card").style.display !== "none") positionCard(s.targetId, s.pad);
    }, true);
    window.addEventListener("resize", () => {
        const s = STEPS[currentStep];
        if (gid("tour-card").style.display !== "none") positionCard(s.targetId, s.pad);
    });

    // Tour is now started by accessTool() when the user enters from the welcome page

    // ── Download button two-line label ────────────────────────────────────────
    (function() {
        function patchBtn() {
            var section = document.getElementById("download_section");
            if (!section) return;
            var btn = (section.tagName === "A" || section.tagName === "BUTTON")
                      ? section
                      : (section.querySelector("a") || section.querySelector("button"));
            if (!btn || btn.querySelector(".dl-header")) return;
            var kids = Array.from(btn.childNodes);
            if (!kids.length) return;
            var fileSpan = document.createElement("span");
            fileSpan.style.cssText = "font-size:0.82em;opacity:0.85;display:block;";
            kids.forEach(function(n) { fileSpan.appendChild(n); });
            btn.appendChild(fileSpan);
            var header = document.createElement("span");
            header.className = "dl-header";
            header.style.cssText = "font-weight:700;font-size:1.1em;display:block;";
            header.textContent = "Download";
            btn.insertBefore(header, btn.firstChild);
            btn.style.setProperty("display",        "flex",      "important");
            btn.style.setProperty("flex-direction", "column",    "important");
            btn.style.setProperty("align-items",    "center",    "important");
            btn.style.setProperty("gap",            "3px",       "important");
            btn.style.setProperty("height",         "auto",      "important");
            btn.style.setProperty("padding",        "10px 20px", "important");
            return true;
        }
        patchBtn();
        var dlBusy = false;
        var dlObs = new MutationObserver(function() {
            if (dlBusy) return; dlBusy = true;
            var patched = patchBtn();
            dlBusy = false;
            if (patched) { dlObs.disconnect(); dlObs = null; }
        });
        dlObs.observe(document.body, { childList: true, subtree: true });
    })();

    // ── MOTION DESIGN ─────────────────────────────────────────────────────────

    // 1. Staggered card entrance
    setTimeout(function() {
        var cards = document.querySelectorAll('#page-main .block, #page-main .form');
        cards.forEach(function(el, i) {
            if (!el.style.animationDelay) el.style.animationDelay = (0.07 * i + 0.06) + 's';
        });
    }, 120);

    // 2. Upload / output corner L-bracket markers — observer disconnects when both panels are found
    (function addCorners() {
        var remaining = 2;
        function inject() {
            ['input_image', 'output_image'].forEach(function(id) {
                var blk = document.getElementById(id);
                if (!blk || blk.dataset.uc) return;
                var wrap = blk.querySelector('[data-testid="image"]');
                if (!wrap) return;
                blk.dataset.uc = '1';
                remaining--;
                if (getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';
                ['tl','tr','bl','br'].forEach(function(c) {
                    var d = document.createElement('div');
                    d.className = 'uc ' + c;
                    wrap.appendChild(d);
                });
            });
            if (remaining <= 0 && obs) { obs.disconnect(); obs = null; }
        }
        var obs = new MutationObserver(function() { inject(); });
        obs.observe(document.body, { childList: true, subtree: true });
        inject();
    })();

    // 3. AI scan line — toggle .is-protecting on the main column while Gradio is busy
    (function watchProgress() {
        var col = document.getElementById('page-main');
        new MutationObserver(function() {
            var busy = !!document.querySelector('[data-testid="progress-bar"], .progress, .eta-bar, .generating');
            if (col) col.classList.toggle('is-protecting', busy);
        }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['class','style'] });
    })();

    // 4. Floating ambient particles (3 only to minimise compositor layers)
    (function spawnParticles() {
        var cfg = [
            { rise:'pRise0', col:'rgba(56,232,255,',  left:'18', size:'2.2', dur:'28', del:'0'  },
            { rise:'pRise2', col:'rgba(124,58,237,',  left:'52', size:'1.8', dur:'34', del:'8'  },
            { rise:'pRise1', col:'rgba(167,139,250,', left:'80', size:'2.0', dur:'30', del:'14' }
        ];
        cfg.forEach(function(c) {
            var p = document.createElement('div');
            var glowR = (parseFloat(c.size) * 2.8).toFixed(1);
            p.style.cssText =
                'position:fixed;bottom:-6px;left:' + c.left + '%;' +
                'width:' + c.size + 'px;height:' + c.size + 'px;border-radius:50%;' +
                'background:' + c.col + '0.85);' +
                'box-shadow:0 0 ' + glowR + 'px ' + c.col + '0.55);' +
                'pointer-events:none;z-index:1;' +
                'animation:' + c.rise + ' ' + c.dur + 's ' + c.del + 's linear infinite;';
            document.body.appendChild(p);
        });
    })();

    // 5. Button click ripple
    (function buttonRipple() {
        document.addEventListener('mousedown', function(e) {
            var btn = e.target.closest('button.primary,button.stop,button.cancel,#protect_btn button,#stop_btn button');
            if (!btn) return;
            var r = btn.getBoundingClientRect();
            var rpl = document.createElement('span');
            rpl.style.cssText =
                'position:absolute;border-radius:50%;width:8px;height:8px;' +
                'margin-top:-4px;margin-left:-4px;background:rgba(255,255,255,0.32);' +
                'pointer-events:none;z-index:9999;' +
                'left:' + (e.clientX - r.left) + 'px;top:' + (e.clientY - r.top) + 'px;' +
                'animation:rippleOut 0.55s ease-out forwards;';
            if (getComputedStyle(btn).position === 'static')
                btn.style.setProperty('position', 'relative', 'important');
            btn.style.overflow = 'hidden';
            btn.appendChild(rpl);
            setTimeout(function() { rpl.remove(); }, 600);
        });
    })();

    // 6. AI shield badge on the output image panel — observer disconnects once injected
    (function injectShield() {
        function add() {
            var outBlk = document.getElementById('output_image');
            if (!outBlk || outBlk.dataset.sv) return;
            var wrap = outBlk.querySelector('[data-testid="image"]');
            if (!wrap) return;
            outBlk.dataset.sv = '1';
            if (getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';
            var sh = document.createElement('div');
            sh.id = 'shield-viz';
            sh.innerHTML =
                '<svg width="48" height="56" viewBox="0 0 22 26" fill="none">' +
                '<path d="M11 1L1 5v8c0 5.5 4.3 10.7 10 12 5.7-1.3 10-6.5 10-12V5L11 1z"' +
                ' stroke="url(#svG)" stroke-width="1.2" fill="rgba(56,232,255,0.04)"/>' +
                '<defs><linearGradient id="svG" x1="0" y1="0" x2="1" y2="1">' +
                '<stop stop-color="#38E8FF"/><stop offset="1" stop-color="#7C3AED"/>' +
                '</linearGradient></defs></svg>';
            sh.style.cssText =
                'position:absolute;bottom:14px;right:14px;pointer-events:none;z-index:4;' +
                'animation:shieldIdlePulse 3.5s ease-in-out infinite;';
            wrap.appendChild(sh);
            if (svObs) { svObs.disconnect(); svObs = null; }
        }
        var svObs = new MutationObserver(function() { add(); });
        svObs.observe(document.body, { childList: true, subtree: true });
        add();
    })();
}
"""

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def _build_css() -> str:
    if BG_IMAGE_PATH.exists():
        body_bg = (
            f"background-image: url('/file={BG_IMAGE_PATH.as_posix()}') !important;"
            " background-size: cover !important;"
            " background-position: center center !important;"
            " background-attachment: fixed !important;"
            " background-repeat: no-repeat !important;"
        )
    else:
        body_bg = (
            "background: radial-gradient(ellipse at 50% 50%, #09112b 0%, #050b1d 40%, #020617 100%) !important;"
            " background-attachment: fixed !important;"
        )
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════════
   LAYER 1 — Gradio CSS variable overrides
   ═══════════════════════════════════════════════════ */
:root {{
    --body-background-fill: #060E1C;
    --body-text-color: #F0F4FF;
    --body-text-color-subdued: #8FA8C8;
    --body-text-size: 0.9rem;
    --font: 'Inter', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

    --block-background-fill: rgba(6,14,28,0.88);
    --block-border-color: rgba(56,232,255,0.30);
    --block-border-width: 1px;
    --block-radius: 14px;
    --block-shadow: 0 0 16px rgba(56,232,255,0.10), 0 0 32px rgba(124,58,237,0.06), 0 6px 28px rgba(0,0,0,0.55);
    --block-padding: 16px;
    --block-title-background-fill: transparent;
    --block-title-border-color: transparent;
    --block-title-text-color: #F0F4FF;

    --block-label-background-fill: rgba(56,232,255,0.07);
    --block-label-border-color: rgba(56,232,255,0.42);
    --block-label-border-width: 1px;
    --block-label-radius: 20px;
    --block-label-padding: 3px 12px;
    --block-label-text-color: #38E8FF;
    --block-label-text-size: 0.72rem;
    --block-label-text-weight: 700;
    --block-label-margin: 0;

    --button-primary-background-fill: linear-gradient(135deg,#00D9FF 0%,#38E8FF 45%,#7C3AED 100%);
    --button-primary-background-fill-hover: linear-gradient(135deg,#18E6FF 0%,#50F0FF 45%,#9B6AF6 100%);
    --button-primary-text-color: #040E1C;
    --button-primary-border-color: transparent;
    --button-primary-border-color-hover: transparent;
    --button-primary-shadow: 0 0 22px rgba(56,232,255,0.38), 0 4px 16px rgba(0,0,0,0.40);
    --button-primary-shadow-hover: 0 0 38px rgba(56,232,255,0.60), 0 8px 28px rgba(0,0,0,0.50);

    --button-cancel-background-fill: linear-gradient(135deg,#FF8C42 0%,#FF5E5B 100%);
    --button-cancel-background-fill-hover: linear-gradient(135deg,#FFAA60 0%,#FF7070 100%);
    --button-cancel-text-color: #ffffff;
    --button-cancel-border-color: transparent;
    --button-cancel-shadow: 0 0 18px rgba(255,94,91,0.28), 0 4px 16px rgba(0,0,0,0.40);

    --button-secondary-background-fill: linear-gradient(135deg,rgba(56,232,255,0.10),rgba(124,58,237,0.10));
    --button-secondary-background-fill-hover: linear-gradient(135deg,rgba(56,232,255,0.22),rgba(124,58,237,0.22));
    --button-secondary-text-color: #38E8FF;
    --button-secondary-border-color: rgba(56,232,255,0.38);
    --button-secondary-border-color-hover: rgba(56,232,255,0.62);

    --button-large-radius: 10px;
    --button-large-text-size: 1rem;
    --button-large-text-weight: 700;
    --button-large-padding: 14px 22px;
    --button-small-radius: 8px;

    --input-background-fill: rgba(3,8,18,0.72);
    --input-border-color: rgba(56,232,255,0.25);
    --input-border-color-focus: rgba(56,232,255,0.65);
    --input-border-width: 1px;
    --input-radius: 9px;
    --input-text-size: 0.9rem;
    --input-text-color: #F0F4FF;
    --input-placeholder-color: #506070;
    --input-shadow-focus: 0 0 10px rgba(56,232,255,0.18);

    --slider-color: #38E8FF;
    --color-accent: #38E8FF;
    --color-accent-soft: rgba(56,232,255,0.15);

    --section-header-text-color: #38E8FF;
    --section-header-text-size: 0.82rem;
    --section-header-text-weight: 700;
    --table-radius: 10px;
    --checkbox-background-color: rgba(3,8,18,0.65);
    --checkbox-border-color: rgba(56,232,255,0.30);
    --checkbox-background-color-selected: #7C3AED;
    --checkbox-border-color-selected: #7C3AED;
}}

/* ═══════════════════════════════════════════════════
   LAYER 2 — CSS selector overrides
   ═══════════════════════════════════════════════════ */

body {{
    {body_bg}
    min-height: 100vh !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}}

body::before {{
    content: '' !important;
    position: fixed !important;
    inset: 0 !important;
    background:
        radial-gradient(ellipse 55% 45% at 82% 88%, rgba(56,232,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 45% 40% at 18% 12%, rgba(124,58,237,0.08) 0%, transparent 60%) !important;
    pointer-events: none !important;
    z-index: 0 !important;
}}

body::after {{
    content: '' !important;
    position: fixed !important;
    inset: 0 !important;
    background-image:
        linear-gradient(rgba(56,232,255,0.017) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,232,255,0.017) 1px, transparent 1px) !important;
    background-size: 50px 50px !important;
    pointer-events: none !important;
    z-index: 0 !important;
}}

.gradio-container {{
    background: transparent !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    max-width: 1350px !important;
    padding: 16px 22px 28px !important;
    position: relative !important;
    z-index: 1 !important;
}}

.block, .form {{
    background: rgba(6,14,28,0.88) !important;
    border: 1px solid rgba(56,232,255,0.30) !important;
    border-radius: 14px !important;
    box-shadow:
        0 0 16px rgba(56,232,255,0.10),
        0 0 32px rgba(124,58,237,0.06),
        0 6px 28px rgba(0,0,0,0.55) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    transition: border-color 0.28s, box-shadow 0.28s !important;
    animation: fadeInUp 0.38s ease both !important;
}}

.block:hover, .form:hover {{
    border-color: rgba(56,232,255,0.50) !important;
    box-shadow:
        0 0 24px rgba(56,232,255,0.18),
        0 0 48px rgba(124,58,237,0.10),
        0 8px 36px rgba(0,0,0,0.62) !important;
}}

#hero-header, #hero-header > .block,
div#hero-header {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important; animation: none !important;
}}

.prose, .prose p, .prose li {{ color: #8FA8C8 !important; }}
.prose h1, .prose h2, .prose h3 {{ color: #F0F4FF !important; font-weight: 700 !important; }}
p, li, span {{ font-family: 'Inter', system-ui, sans-serif !important; }}
strong {{ color: #F0F4FF !important; }}
hr {{ border-color: rgba(56,232,255,0.16) !important; }}

.block .label-wrap, .form .label-wrap,
.block-label, .block > div > .label-wrap {{
    display: inline-flex !important;
    align-items: center !important;
    background: rgba(56,232,255,0.07) !important;
    border: 1px solid rgba(56,232,255,0.42) !important;
    border-radius: 20px !important;
    padding: 3px 12px !important;
    margin-bottom: 10px !important;
    width: auto !important;
    max-width: max-content !important;
    min-width: unset !important;
}}

.block .label-wrap span, .block .label-wrap label span,
.block-label span {{
    color: #38E8FF !important;
    -webkit-text-fill-color: #38E8FF !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}}

input[type=text], input[type=number], input[type=email], textarea, select {{
    background: rgba(3,8,18,0.72) !important;
    border: 1px solid rgba(56,232,255,0.25) !important;
    border-radius: 9px !important;
    color: #F0F4FF !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}}

input[type=text]:focus, input[type=number]:focus, textarea:focus {{
    border-color: rgba(56,232,255,0.65) !important;
    box-shadow: 0 0 10px rgba(56,232,255,0.20) !important;
    outline: none !important;
}}

.info, .description {{ color: #5E7890 !important; font-size: 0.78rem !important; }}

[data-testid="image"], .image-container {{
    background: rgba(3,8,18,0.65) !important;
    border: 1px solid rgba(56,232,255,0.28) !important;
    border-radius: 12px !important;
    transition: border-color 0.28s, box-shadow 0.28s !important;
}}

[data-testid="image"]:hover, .image-container:hover {{
    border-color: rgba(56,232,255,0.52) !important;
    box-shadow: 0 0 20px rgba(56,232,255,0.12) !important;
}}

button.primary, .primary button,
#protect_btn button, #protect_btn .lg {{
    background: linear-gradient(135deg,#00D9FF 0%,#38E8FF 45%,#7C3AED 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #040E1C !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    min-height: 52px !important;
    box-shadow: 0 0 22px rgba(56,232,255,0.38), 0 4px 16px rgba(0,0,0,0.42) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease !important;
}}

button.primary:hover, #protect_btn button:hover, #protect_btn .lg:hover {{
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 38px rgba(56,232,255,0.60), 0 8px 28px rgba(0,0,0,0.52) !important;
    filter: brightness(1.08) !important;
}}

button.primary:active {{ transform: scale(0.98) !important; }}

button.stop, button.cancel, .stop button, .cancel button,
#stop_btn button, #stop_btn .lg {{
    background: linear-gradient(135deg,#FF8C42 0%,#FF5E5B 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    min-height: 52px !important;
    box-shadow: 0 0 18px rgba(255,94,91,0.28), 0 4px 16px rgba(0,0,0,0.42) !important;
    transition: transform 0.22s ease, box-shadow 0.22s ease !important;
}}

button.stop:hover, #stop_btn button:hover, #stop_btn .lg:hover {{
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 34px rgba(255,94,91,0.52), 0 8px 28px rgba(0,0,0,0.52) !important;
}}

button.secondary, .secondary button,
#download_section a, #download_section button,
.download_section a, .download_section button {{
    background: linear-gradient(135deg,rgba(56,232,255,0.10),rgba(124,58,237,0.10)) !important;
    border: 1px solid rgba(56,232,255,0.38) !important;
    border-radius: 10px !important;
    color: #38E8FF !important;
    font-weight: 600 !important;
    min-height: 50px !important;
    transition: background 0.28s, box-shadow 0.28s, transform 0.2s !important;
}}

#download_section a:hover, #download_section button:hover {{
    background: linear-gradient(135deg,rgba(56,232,255,0.20),rgba(124,58,237,0.20)) !important;
    box-shadow: 0 0 22px rgba(56,232,255,0.24) !important;
    transform: translateY(-1px) !important;
}}

input[type=radio] {{ accent-color: #7C3AED !important; }}

.gradio-radio .wrap, fieldset .wrap {{
    background: rgba(3,8,18,0.62) !important;
    border: 1px solid rgba(56,232,255,0.20) !important;
    border-radius: 8px !important;
    padding: 7px 14px !important;
    transition: all 0.22s !important;
    cursor: pointer !important;
    margin: 2px 4px 2px 0 !important;
}}

.gradio-radio .wrap:has(input:checked) {{
    background: rgba(124,58,237,0.22) !important;
    border-color: rgba(124,58,237,0.68) !important;
    box-shadow: 0 0 12px rgba(124,58,237,0.30) !important;
}}

.gradio-radio .wrap:has(input:checked) span {{
    color: #F0F4FF !important;
    -webkit-text-fill-color: #F0F4FF !important;
}}

input[type=range] {{
    -webkit-appearance: none !important;
    appearance: none !important;
    background: linear-gradient(90deg,#38E8FF 0%,#7C3AED 100%) !important;
    height: 4px !important;
    border-radius: 4px !important;
    border: none !important;
}}

input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none !important;
    width: 18px !important; height: 18px !important;
    border-radius: 50% !important;
    background: #F0F4FF !important;
    box-shadow: 0 0 8px rgba(56,232,255,0.75), 0 0 16px rgba(124,58,237,0.45) !important;
    cursor: pointer !important;
    transition: box-shadow 0.22s !important;
}}

input[type=range]::-webkit-slider-thumb:hover {{
    box-shadow: 0 0 14px rgba(56,232,255,1.0), 0 0 26px rgba(124,58,237,0.75) !important;
}}

input[type=range]::-moz-range-thumb {{
    width: 18px !important; height: 18px !important;
    border-radius: 50% !important; border: none !important;
    background: #F0F4FF !important;
    box-shadow: 0 0 8px rgba(56,232,255,0.75), 0 0 16px rgba(124,58,237,0.45) !important;
    cursor: pointer !important;
}}

#TxtGPU textarea {{
    background: rgba(56,232,255,0.04) !important;
    border-color: rgba(56,232,255,0.26) !important;
    color: #38E8FF !important;
    font-size: 0.82rem !important;
}}

.upload-container span, .file-preview-holder span,
[data-testid="image"] .wrap span {{ color: #8FA8C8 !important; }}

/* About & Guide pages — hidden by default; JS sets display:flex on show */
#page-about, #page-guide {{
    display: none;
    max-width: 860px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    flex-direction: column !important;
    gap: 12px !important;
}}

/* About & Guide pages — typography */
#page-about .prose h1, #page-about .prose .h1,
#page-guide .prose h1, #page-guide .prose .h1 {{
    font-size: 1.72rem !important;
    font-weight: 800 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    letter-spacing: 0.02em !important;
    background: linear-gradient(90deg, #38E8FF 0%, #A78BFA 60%, #6EE7FF 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    line-height: 1.35 !important;
    margin-bottom: 0.5rem !important;
}}
#page-about .prose h2, #page-about .prose .h2,
#page-guide .prose h2, #page-guide .prose .h2 {{
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #38E8FF !important;
    -webkit-text-fill-color: #38E8FF !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.45rem !important;
}}
#page-about .prose h3, #page-about .prose .h3,
#page-guide .prose h3, #page-guide .prose .h3 {{
    font-size: 0.97rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #A78BFA !important;
    -webkit-text-fill-color: #A78BFA !important;
    letter-spacing: 0.04em !important;
    margin-top: 1rem !important;
    margin-bottom: 0.35rem !important;
}}
#page-about .prose, #page-about .prose p, #page-about .prose li,
#page-about .prose ul, #page-about .prose ol,
#page-guide .prose, #page-guide .prose p, #page-guide .prose li,
#page-guide .prose ul, #page-guide .prose ol {{
    color: #8FA8C8 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    -webkit-text-fill-color: #8FA8C8 !important;
}}
#page-about .prose strong, #page-about strong,
#page-guide .prose strong, #page-guide strong {{
    color: #F0F4FF !important;
    -webkit-text-fill-color: #F0F4FF !important;
    font-weight: 700 !important;
}}
#page-about .prose u, #page-guide .prose u {{
    text-decoration-color: rgba(56,232,255,0.55) !important;
    text-underline-offset: 3px !important;
}}
#page-about .prose hr, #page-about hr,
#page-guide .prose hr, #page-guide hr {{
    border: none !important;
    border-top: 1px solid rgba(56,232,255,0.16) !important;
    margin: 1.4rem 0 !important;
}}
#page-about .prose a, #page-about a:not([onclick]),
#page-guide .prose a, #page-guide a:not([onclick]) {{
    color: #38E8FF !important;
    -webkit-text-fill-color: #38E8FF !important;
    text-decoration: underline !important;
    text-decoration-color: rgba(56,232,255,0.40) !important;
    text-underline-offset: 3px !important;
}}
#page-about .prose a:hover, #page-about a:not([onclick]):hover,
#page-guide .prose a:hover, #page-guide a:not([onclick]):hover {{
    color: #6EE7FF !important;
    -webkit-text-fill-color: #6EE7FF !important;
    text-decoration-color: rgba(110,231,255,0.65) !important;
}}
#page-about .prose code, #page-about code,
#page-guide .prose code, #page-guide code {{
    color: #38E8FF !important;
    -webkit-text-fill-color: #38E8FF !important;
    background: rgba(56,232,255,0.10) !important;
    border: 1px solid rgba(56,232,255,0.20) !important;
    padding: 1px 6px !important;
    border-radius: 5px !important;
    font-size: 0.85em !important;
}}

/* Glass cards inside the reading column */
#page-about .block, #page-guide .block {{
    padding: 2.2rem 2.8rem 2.4rem !important;
    border-radius: 18px !important;
}}

#page-about .prose, #page-guide .prose {{
    padding: 0 !important;
    margin: 0 !important;
}}

#page-about .prose p, #page-guide .prose p {{
    margin-top: 0 !important;
    margin-bottom: 0.85rem !important;
}}
#page-about .prose li, #page-guide .prose li {{
    margin-bottom: 0.4rem !important;
}}
#page-about .prose ul, #page-about .prose ol,
#page-guide .prose ul, #page-guide .prose ol {{
    padding-left: 1.4rem !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.85rem !important;
}}

#page-about .prose h1:first-child, #page-about .prose > h1:first-of-type,
#page-guide .prose h1:first-child, #page-guide .prose > h1:first-of-type {{
    margin-top: 0 !important;
    padding-top: 0 !important;
}}

#page-about .block:last-child, #page-guide .block:last-child {{
    padding: 1.2rem 2.8rem !important;
}}

@media (max-width: 960px) {{
    #page-about .block, #page-guide .block {{
        padding: 1.8rem 1.8rem 2rem !important;
    }}
    #page-about .block:last-child, #page-guide .block:last-child {{
        padding: 1rem 1.8rem !important;
    }}
}}

@media (max-width: 640px) {{
    #page-about, #page-guide {{
        max-width: 100% !important;
        gap: 8px !important;
    }}
    #page-about .block, #page-guide .block {{
        padding: 1.4rem 1.2rem 1.6rem !important;
        border-radius: 12px !important;
    }}
    #page-about .block:last-child, #page-guide .block:last-child {{
        padding: 0.9rem 1.2rem !important;
    }}
}}

/* User Guide — raw HTML section styling */
.guide-section {{ margin-bottom: 28px; }}
.guide-section h2 {{
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    color: #38E8FF !important;
    -webkit-text-fill-color: #38E8FF !important;
    margin: 0 0 8px !important;
    border-bottom: 1px solid rgba(56,232,255,0.20) !important;
    padding-bottom: 4px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}}
.guide-section p, .guide-section li {{
    font-size: 0.9rem !important;
    line-height: 1.75 !important;
    color: #8FA8C8 !important;
    -webkit-text-fill-color: #8FA8C8 !important;
    margin: 0 0 6px !important;
}}
.guide-section ul {{ padding-left: 20px !important; margin: 0 0 6px !important; }}
.guide-tag {{
    display: inline-block !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    margin-right: 6px !important;
}}
.tag-ip2p {{ background: rgba(56,232,255,0.15) !important; color: #38E8FF !important; -webkit-text-fill-color: #38E8FF !important; }}
.tag-sd   {{ background: rgba(167,139,250,0.15) !important; color: #A78BFA !important; -webkit-text-fill-color: #A78BFA !important; }}
.tag-warn {{ background: rgba(255,180,50,0.12) !important;  color: #FFB432 !important; -webkit-text-fill-color: #FFB432 !important; }}
.tag-rec  {{ background: rgba(56,232,100,0.12) !important;  color: #56E864 !important; -webkit-text-fill-color: #56E864 !important; }}

.rec-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 10px; }}
.rec-table th {{
    background: rgba(56,232,255,0.10) !important;
    color: #38E8FF !important;
    -webkit-text-fill-color: #38E8FF !important;
    padding: 8px 10px;
    text-align: left;
    font-weight: 700;
    border-bottom: 1px solid rgba(56,232,255,0.28) !important;
}}
.rec-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid rgba(56,232,255,0.07) !important;
    color: #8FA8C8 !important;
    -webkit-text-fill-color: #8FA8C8 !important;
}}
.rec-table tr:nth-child(even) td {{ background: rgba(56,232,255,0.03) !important; }}
.rec-table tr:hover td {{ background: rgba(56,232,255,0.06) !important; }}
.rec-table .warn-row td {{
    color: #FFB432 !important;
    -webkit-text-fill-color: #FFB432 !important;
    background: rgba(255,180,50,0.05) !important;
}}

/* Footer */
footer {{ display: none !important; }}

/* Progress bar */
.progress-bar, .progress {{
    background: linear-gradient(90deg,#38E8FF,#7C3AED) !important;
    border-radius: 4px !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: rgba(3,8,18,0.6); border-radius: 3px; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg,#38E8FF,#7C3AED); border-radius: 3px; }}

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes borderPulse {{
    0%,100% {{ box-shadow:0 0 16px rgba(56,232,255,0.10),0 0 32px rgba(124,58,237,0.06),0 6px 28px rgba(0,0,0,0.55); }}
    50%     {{ box-shadow:0 0 28px rgba(56,232,255,0.24),0 0 52px rgba(124,58,237,0.14),0 6px 28px rgba(0,0,0,0.55); }}
}}
.block, .form {{
    animation: fadeInUp 0.42s ease both !important;
}}

@keyframes protectGlow {{
    0%,100% {{ box-shadow:0 0 22px rgba(56,232,255,0.38),0 4px 16px rgba(0,0,0,0.42); }}
    50%     {{ box-shadow:0 0 46px rgba(56,232,255,0.74),0 0 26px rgba(124,58,237,0.42),0 6px 20px rgba(0,0,0,0.44); }}
}}
button.primary, #protect_btn button, #protect_btn .lg {{
    box-shadow: 0 0 22px rgba(56,232,255,0.38), 0 4px 16px rgba(0,0,0,0.42) !important;
}}
button.primary:hover, #protect_btn button:hover, #protect_btn .lg:hover {{
    animation: none !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 0 52px rgba(56,232,255,0.82),0 8px 28px rgba(0,0,0,0.52) !important;
    filter: brightness(1.08) !important;
}}

@keyframes navSlideDown {{
    from {{ opacity:0; transform:translateY(-20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
#main-nav {{ animation: navSlideDown 0.55s cubic-bezier(0.16,1,0.3,1) both !important; }}

@keyframes sliderShimmer {{
    from {{ background-position:0% 50%; }}
    to   {{ background-position:100% 50%; }}
}}
input[type=range] {{
    background: linear-gradient(90deg,#7C3AED 0%,#38E8FF 40%,#A78BFA 100%) !important;
    height: 4px !important;
    border-radius: 4px !important;
    border: none !important;
}}

@keyframes cornerBlink {{
    0%,100% {{ opacity:0.38; }}
    50%     {{ opacity:1; }}
}}
.uc {{
    position:absolute; width:14px; height:14px; pointer-events:none; z-index:3;
    animation: cornerBlink 2.4s ease-in-out infinite;
}}
.uc::before, .uc::after {{ content:''; position:absolute; background:#38E8FF; border-radius:1px; }}
.uc::before {{ width:2px; height:14px; top:0; left:0; }}
.uc::after  {{ width:14px; height:2px; top:0; left:0; }}
.uc.tl {{ top:8px; left:8px; }}
.uc.tr {{ top:8px; right:8px; transform:scaleX(-1); }}
.uc.bl {{ bottom:8px; left:8px; transform:scaleY(-1); }}
.uc.br {{ bottom:8px; right:8px; transform:scale(-1); animation-delay:0.6s; }}

@keyframes scanDown {{
    0%   {{ top:0%;opacity:0; }}
    6%   {{ opacity:1; }}
    94%  {{ opacity:1; }}
    100% {{ top:100%;opacity:0; }}
}}
.is-protecting [data-testid="image"],
.is-protecting .image-container {{
    position: relative !important; overflow: hidden !important;
}}
.is-protecting [data-testid="image"]::after,
.is-protecting .image-container::after {{
    content: '' !important; position: absolute !important;
    left:0; right:0; top:0; height:3px !important;
    background: linear-gradient(90deg,transparent 0%,#38E8FF 35%,#A78BFA 65%,transparent 100%) !important;
    animation: scanDown 2.5s linear infinite !important;
    pointer-events: none !important; z-index:10 !important;
}}

@keyframes heroShimmer {{
    from {{ background-position:0% 50%; }}
    to   {{ background-position:100% 50%; }}
}}

@keyframes af1 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(34px,-46px) scale(1.06); }} }}
@keyframes af2 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(-30px,38px) scale(0.95); }} }}
@keyframes af3 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(46px,28px) scale(1.04); }} }}
@keyframes af4 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(-36px,-26px) scale(0.97); }} }}
@keyframes af5 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(22px,44px) scale(1.03); }} }}

@keyframes pRise0 {{ 0%{{transform:translateY(0) translateX(0);opacity:0}} 8%{{opacity:1}} 92%{{opacity:1}} 100%{{transform:translateY(-105vh) translateX(-65px);opacity:0}} }}
@keyframes pRise1 {{ 0%{{transform:translateY(0) translateX(0);opacity:0}} 8%{{opacity:1}} 92%{{opacity:1}} 100%{{transform:translateY(-105vh) translateX(48px);opacity:0}} }}
@keyframes pRise2 {{ 0%{{transform:translateY(0) translateX(0);opacity:0}} 8%{{opacity:1}} 92%{{opacity:1}} 100%{{transform:translateY(-105vh) translateX(-28px);opacity:0}} }}
@keyframes pRise3 {{ 0%{{transform:translateY(0) translateX(0);opacity:0}} 8%{{opacity:1}} 92%{{opacity:1}} 100%{{transform:translateY(-105vh) translateX(72px);opacity:0}} }}

@keyframes rippleOut {{ 0%{{transform:scale(1);opacity:0.55}} 100%{{transform:scale(24);opacity:0}} }}

@keyframes shieldIdlePulse {{
    0%,100% {{ opacity:0.55; filter:drop-shadow(0 0 4px rgba(56,232,255,0.40)); }}
    50%     {{ opacity:0.90; filter:drop-shadow(0 0 10px rgba(56,232,255,0.80)) drop-shadow(0 0 6px rgba(124,58,237,0.50)); }}
}}

@keyframes bsFloat1 {{ 0%,100% {{ transform:translate(0,0) rotate(-8deg);  }} 50% {{ transform:translate(18px,-28px) rotate(-5deg);  }} }}
@keyframes bsFloat2 {{ 0%,100% {{ transform:translate(0,0) rotate(12deg);  }} 50% {{ transform:translate(-22px,18px) rotate(10deg);  }} }}
@keyframes bsFloat3 {{ 0%,100% {{ transform:translate(0,0) rotate(-3deg);  }} 50% {{ transform:translate(14px,22px) rotate(0deg);    }} }}
@keyframes bsFloat4 {{ 0%,100% {{ transform:translate(0,0) rotate(7deg);   }} 50% {{ transform:translate(-14px,-18px) rotate(10deg); }} }}
@keyframes bsFloat5 {{ 0%,100% {{ transform:translate(0,0) rotate(-14deg); }} 50% {{ transform:translate(16px,14px) rotate(-11deg);  }} }}
@keyframes bgOrbSpin1 {{ from {{ transform:rotate(0deg);   }} to {{ transform:rotate(360deg);  }} }}
@keyframes bgOrbSpin2 {{ from {{ transform:rotate(0deg);   }} to {{ transform:rotate(-360deg); }} }}

@keyframes tourPulse {{
    0%,74%,100% {{ transform:scale(1) translateY(0);      box-shadow:0 0 18px rgba(56,232,255,0.42),0 4px 14px rgba(0,0,0,0.4); }}
    80%         {{ transform:scale(1.06) translateY(-2px); box-shadow:0 0 36px rgba(56,232,255,0.82),0 0 20px rgba(139,92,246,0.48),0 6px 18px rgba(0,0,0,0.5); }}
}}
#tour-reopen:hover {{ transform: translateY(-2px) !important; filter: brightness(1.12) !important; }}

/* Collapse Gradio wrappers for elements moved to body by JS */
.gradio-container > .block:has(> #page-welcome),
.gradio-container > .block:has(> #tour-root) {{
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }}
}}

.gradio-row {{ gap: 16px !important; align-items: stretch !important; }}
.gradio-column {{ gap: 14px !important; }}
"""


CUSTOM_CSS = _build_css()

with gr.Blocks() as demo:
    gr.HTML(NAV_HTML)
    gr.HTML(TOUR_HTML)
    gr.HTML(WELCOME_PAGE_HTML)

    # ── Main page ────────────────────────────────────────────────────────────
    with gr.Column(elem_id="page-main"):
        gr.Markdown(
            """
            # SafeShot: A Tool to Protect Personal Images

            Upload an image and obtain a protected version of it to resist
            Image-Generation diffusion models.
            """,
            elem_id="hero-header",
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Your Image",
                                       elem_id="input_image")

                mode = gr.Radio(
                    choices=["IP2P", "SD"],
                    value="IP2P",
                    label="Protection Mode",
                    info=(
                        "IP2P: targets InstructPix2Pix editors (EditShield). "
                        "SD: targets Stable Diffusion pipelines (BlurGuard)."
                    ),
                    elem_id="mode_selector",
                )

                resolution = gr.Radio(
                    choices=["Original", 128, 256, 512],
                    value="Original",
                    label="Output Resolution",
                    info="Original keeps the source image dimensions. Lower fixed sizes process faster.",
                    elem_id="resolution_selector",
                )

                eps_slider = gr.Slider(
                    minimum=_EPS_MIN_255,
                    maximum=_EPS_MAX_255,
                    value=_EPS_IP2P_255,
                    step=1,
                    label="Perturbation Strength (ε, in units of 1/255)",
                    info=(
                        f"Controls the maximum pixel change added for protection. "
                        f"Default: {_EPS_IP2P_255}/255 for IP2P, {_EPS_SD_255}/255 for SD."
                    ),
                    elem_id="eps_slider",
                )

                steps = gr.Slider(
                    minimum=20,
                    maximum=100,
                    value=20,
                    step=10,
                    label="Optimization Steps",
                    info="More steps may strengthen protection but take longer.",
                    elem_id="steps_slider",
                )

                with gr.Row():
                    protect_btn = gr.Button(
                        "Protect Image",
                        variant="primary",
                        elem_id="protect_btn",
                    )
                    stop_btn = gr.Button(
                        "Stop",
                        variant="stop",
                        elem_id="stop_btn",
                    )
                text_output = gr.Textbox(label="Device Status", visible=False,
                                         elem_id="TxtGPU")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Protected Image",
                                        elem_id="output_image")
                output_file = gr.DownloadButton(label="Download Protected Image", variant="primary",
                                      visible=False, elem_id="download_section", elem_classes="download_section")

        original_state = gr.State(None)

        # When mode changes, update the eps slider default
        def update_eps_default(selected_mode):
            default = _EPS_IP2P_255 if selected_mode == "IP2P" else _EPS_SD_255
            return gr.update(value=default)

        mode.change(fn=update_eps_default, inputs=mode, outputs=eps_slider)

        protect_btn.click(
            fn=prereq_gpu,
            inputs=input_image,
            outputs=text_output,
            queue=False,
        )
        protection_event = protect_btn.click(
            fn=model,
            inputs=[input_image, mode, resolution, eps_slider, steps],
            outputs=[output_image, output_file],
        )
        stop_btn.click(
            fn=stop_protection,
            outputs=text_output,
            cancels=[protection_event],
            queue=False,
        )

        # Resolution preview — store original on upload, show resized preview on resolution change
        def on_image_upload(image, current_resolution):
            return image, _preview_resize(image, current_resolution)

        # Use .upload (not .change) so programmatic updates to input_image don't retrigger this
        input_image.upload(
            fn=on_image_upload,
            inputs=[input_image, resolution],
            outputs=[original_state, input_image],
            queue=False,
        )
        # When user clears the image, reset stored original
        input_image.clear(
            fn=lambda: None,
            outputs=original_state,
            queue=False,
        )
        resolution.change(
            fn=_preview_resize,
            inputs=[original_state, resolution],
            outputs=input_image,
            queue=False,
        )

    # ── User Guide page ──────────────────────────────────────────────────────
    with gr.Column(elem_id="page-guide"):
        gr.Markdown("# User Guide")
        gr.HTML("""
<style>
  .guide-section { margin-bottom: 28px; }
  .guide-section h2 {
    font-size: 17px; font-weight: 700; color: #1e40af;
    margin: 0 0 8px; border-bottom: 2px solid #dbeafe; padding-bottom: 4px;
  }
  .guide-section p, .guide-section li {
    font-size: 14px; line-height: 1.7; color: #374151; margin: 0 0 6px;
  }
  .guide-section ul { padding-left: 20px; margin: 0 0 6px; }
  .guide-tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; font-weight: 700; margin-right: 6px;
  }
  .tag-ip2p { background: #dbeafe; color: #1e40af; }
  .tag-sd   { background: #fce7f3; color: #9d174d; }
  .tag-warn { background: #fef9c3; color: #854d0e; }
  .tag-rec  { background: #dcfce7; color: #166534; }
  .strength-demo-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px; margin: 14px 0 8px;
  }
  .strength-demo-card {
    margin: 0; border: 1px solid #e5e7eb; border-radius: 8px;
    overflow: hidden; background: #fff;
  }
  .strength-demo-card img {
    display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover;
    background: #f8fafc;
  }
  .strength-demo-card figcaption {
    padding: 8px 9px; min-height: 56px;
  }
  .strength-demo-card strong {
    display: block; color: #111827; font-size: 13px; margin-bottom: 2px;
  }
  .strength-demo-card span {
    display: block; color: #4b5563; font-size: 12px; line-height: 1.35;
  }
  .strength-demo-note {
    font-size: 13px !important; color: #4b5563 !important; margin-top: 8px !important;
  }
  @media (max-width: 760px) {
    .strength-demo-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }

  /* Recommendation table */
  .rec-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
  .rec-table th {
    background: #1e40af; color: white; padding: 8px 10px;
    text-align: left; font-weight: 700;
  }
  .rec-table td { padding: 7px 10px; border-bottom: 1px solid #e5e7eb; color: #374151; }
  .rec-table tr:nth-child(even) td { background: #f8fafc; }
  .rec-table tr:hover td { background: #eff6ff; }
  .rec-table .warn-row td { color: #92400e; background: #fffbeb; }
</style>

<!-- Protection Mode -->
<div class="guide-section">
  <h2>Protection Mode</h2>
  <p>Choose the mode that matches the type of AI editor you want to protect against.</p>
  <ul>
    <li>
      <span class="guide-tag tag-ip2p">IP2P</span>
      <strong>InstructPix2Pix (EditShield)</strong> — Designed to disrupt instruction-based image editors
      such as models that accept a text prompt like "make the person smile" or "turn into an oil painting."
      It confuses the VAE (image encoder) that these models use internally.
      Default &epsilon; = 4/255. Fastest mode and typically less visible.
    </li>
    <li style="margin-top:8px;">
      <span class="guide-tag tag-sd">SD</span>
      <strong>Stable Diffusion (BlurGuard)</strong> — Designed to disrupt Stable Diffusion inpainting pipelines.
      Runs in two stages: first it learns adaptive blur kernels for each region of the image (50 warmup steps),
      then it pushes the image's latent representation toward a random unpredictable target.
      The result is a low-frequency perturbation that survives JPEG compression and purification attempts.
      Default &epsilon; = 16/255. Slower and requires more steps than IP2P mode.
      <br><strong>Note:</strong> the warmup phase automatically scales to roughly one-third of your chosen step count,
      so Stage 2 (the actual PGD attack) always receives the remaining two-thirds of steps regardless of how many you select.
    </li>
  </ul>
</div>


<!-- Optimization Steps -->
<div class="guide-section">
  <h2>Optimization Steps</h2>
  <p>
    Each step is one optimization update used to compute the protection layer. More steps usually
    give the optimizer more opportunity to improve the perturbation, but the benefit may plateau
    and runtime will increase.
  </p>
  <ul>
    <li><strong>20 steps</strong> — Fast preview. Useful for testing the workflow or running on slower hardware, but protection may be weaker.</li>
    <li><strong>50 steps</strong> — Medium setting. A reasonable starting point for local testing when balancing speed and protection quality.</li>
    <li><strong>80 steps</strong> — Stronger setting. Better for final outputs when runtime is acceptable.</li>
    <li><strong>100 steps</strong> — Highest setting in this app. Usually gives the strongest optimization result, but takes the longest and may show diminishing returns.</li>
  </ul>
  <p>
    <strong>Note:</strong> More steps do not guarantee perfect protection. The final result also depends on the selected mode,
    image resolution, epsilon/strength, hardware, and the target editing model.
  </p>
</div>


<!-- Output Resolution -->
<div class="guide-section">
  <h2>Output Resolution</h2>
  <p>Controls the size of the image that is fed into the protection model and returned to you.</p>
  <ul>
    <li><strong>Original</strong> (default) — The image is kept at its original dimensions (rounded down to the nearest multiple of 8 for technical compatibility). The output will be the same size as your input.</li>
    <li><strong>128 / 256 / 512</strong> — Your image is resized and center-cropped to a square of that pixel size before protection runs. Use lower values on slow hardware to reduce runtime.</li>
  </ul>
  <p>
    <span class="guide-tag tag-warn">Note</span>
    Smaller resolution = faster, but fine details are lost during resizing. For portrait photos,
    256&times;256 or 512&times;512 preserves the most identity-relevant detail.
  </p>
</div>

<!-- Epsilon -->
<div class="guide-section">
  <h2>Perturbation Strength (&epsilon;, in units of 1/255)</h2>
  <p>
    Epsilon is the maximum change allowed per pixel channel. Higher values create a stronger,
    harder-to-remove protection layer at the cost of slightly more visible noise.
  </p>
  <ul>
    <li><strong>&epsilon; = 8/255</strong> — Barely visible; commonly used in research benchmarks. Default for IP2P mode.</li>
    <li><strong>&epsilon; = 16/255</strong> — Noticeably stronger noise; more robust against purification attacks. Default for SD mode.</li>
    <li><strong>&epsilon; = 24–32/255</strong> — Visible to the naked eye; use only when maximum robustness is needed.</li>
  </ul>
  <p>
    <span class="guide-tag tag-warn">Note</span>
    When you switch between IP2P and SD mode, the slider resets to the recommended default for that mode automatically.
    You can always override it manually.
  </p>
</div>
""" + _demo_strength_gallery_html() + """

<!-- Recommendation table -->
<div class="guide-section">
  <h2>Recommended Parameters by Hardware</h2>
  <p>
    Use this table as a starting point. Times are rough estimates after the models are already loaded.
    The first run may take longer because of model loading, setup, or cache initialization. Later runs
    in the same app window may be faster. Actual speed also depends on memory pressure, background
    processes, thermal throttling, and the selected protection mode.
  </p>

  <table class="rec-table">
    <thead>
      <tr>
        <th>Hardware</th>
        <th>Mode</th>
        <th>Resolution</th>
        <th>Steps</th>
        <th>&epsilon;</th>
        <th>Estimated Time</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="3"><strong>Apple M1 / M2 / M3<br>(MPS)</strong></td>
        <td><span class="guide-tag tag-ip2p">IP2P</span></td>
        <td>512 px</td>
        <td>30–60</td>
        <td>4/255</td>
        <td>~2 min</td>
      </tr>
      <tr>
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td>256 px</td>
        <td>100</td>
        <td>16/255</td>
        <td>~1–2 min</td>
      </tr>
      <tr class="warn-row">
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td>512 px</td>
        <td>40</td>
        <td>16/255</td>
        <td>~25 min; use only for final tests</td>
      </tr>

      <tr>
        <td rowspan="3"><strong>Nvidia GPU<br>4–6 GB VRAM</strong></td>
        <td><span class="guide-tag tag-ip2p">IP2P</span></td>
        <td>512 px</td>
        <td>60–100</td>
        <td>4/255</td>
        <td>~1–3 min</td>
      </tr>
      <tr>
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td>256 px</td>
        <td>100</td>
        <td>16/255</td>
        <td>~30–90 sec</td>
      </tr>
      <tr>
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td>512 px</td>
        <td>40</td>
        <td>16/255</td>
        <td>~8–20 min, depending on VRAM/offload</td>
      </tr>

      <tr>
        <td rowspan="3"><strong>Nvidia GPU<br>8+ GB VRAM</strong></td>
        <td><span class="guide-tag tag-ip2p">IP2P</span></td>
        <td>512 px</td>
        <td>100</td>
        <td>4/255</td>
        <td>~1–2 min</td>
      </tr>
      <tr>
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td>256 px</td>
        <td>100</td>
        <td>16/255</td>
        <td>~20–60 sec</td>
      </tr>
      <tr>
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td>512 px</td>
        <td>40–100</td>
        <td>16/255</td>
        <td>~5–15 min</td>
      </tr>

      <tr class="warn-row">
        <td><strong>CPU only<br>(no GPU)</strong></td>
        <td><span class="guide-tag tag-ip2p">IP2P</span></td>
        <td>128–256 px</td>
        <td>20</td>
        <td>4/255</td>
        <td>Slow; testing only</td>
      </tr>
      <tr class="warn-row">
        <td></td>
        <td><span class="guide-tag tag-sd">SD</span></td>
        <td colspan="3" style="color:#b91c1c;">Not recommended — use GPU or MPS if available</td>
        <td>—</td>
      </tr>
    </tbody>
  </table>

  <p style="margin-top:10px;">
    <span class="guide-tag tag-rec">Tip</span>
    Start with 256 px to confirm the tool works on your machine, then increase resolution only for
    final outputs. SD mode at 512 px can be much slower than 256 px, especially on MacBook Air or
    low-VRAM GPUs.
  </p>

  <p style="margin-top:8px;">
    <span class="guide-tag tag-rec">Observed example</span>
    On a MacBook Air M2, IP2P at 512 px with 30–60 steps took about 100 seconds, SD at 256 px with
    100 steps took about 36 seconds, while SD at 512 px with 40 steps took about 25 minutes.
  </p>
</div>


""")
        gr.HTML(
            """
            <div style="margin-top:8px;">
              <a onclick="window.showPage('main')"
                 style="cursor:pointer; display:inline-block; padding:10px 22px;
                        background:#2563eb; color:white; border-radius:8px;
                        font-weight:600; font-size:14px; text-decoration:none;">
                &larr; Back to the Tool
              </a>
            </div>
            """
        )

    # ── About page ───────────────────────────────────────────────────────────
    with gr.Column(elem_id="page-about"):
        gr.Markdown(
            """
            # About This Tool

            ---

            ## Project context

            With the increase of capacity in AI models, many cases of misusage have been reported,
            one of which is the generation of harmful images, known as deepfakes, specifically with diffusion models
            such as Stable Diffusion or DALL·E.

            As part of the AI4Good Lab, our team wants to create a project to mitigate this issue, by trying to prevent these
            generations in the first place. Our team gathered resources on current available models that can add a protection layer
            on top of your original personal image-so that diffusion models wouldn't be able to edit it when prompted-and we
            came up with a user-friendly product for everyone to use.

            ---

            ## What does it do?

            Using adversarial pertubation, our tool will add a protection layer that consists of pixel-level noise, computed just enough
            so that the change is small enough to remain undetected with the human eye, but big enough to disrupt an Image-Generation process by a
            diffusion model. You can then download the protected version of your image.

            The goal is to confuse the AI when it's following a malicious prompt. If anyone now tries to feed your protected image into an AI diffusion model
            in order to edit it, the output result would look unrealistic, defeating the malicious intent.

            **<u>IMPORTANT:</u>** We can't promise universal protection. Our project goal is to raise the cost of
            malicious image editing, and offer a mitigating solution for everyone to use. Our model is tested on white-box diffusion models, and has 
            no promising effect on black-box models such as DALLE-3.

            ---

            ## How does it work?

            Our tool has two protection modes, each designed to disrupt a different type of AI editor:

            - **IP2P Mode (EditShield)** — Targeted at InstructPix2Pix-based editors. Adds adversarial perturbations that maximally confuse the VAE latent representation used by instruction-following diffusion models. Use this if you are worried about text-prompt-based image editing.

            - **SD Mode (BlurGuard)** — Targeted at Stable Diffusion inpainting pipelines. Runs a two-stage attack: first it learns per-region blur kernels (SLIC segmentation + frequency-aware warmup), then pushes the image toward a random point in the VAE latent space. Perturbations are low-frequency and robust to JPEG compression and purification attempts.

            **<u>IMPORTANT:</u>** We didn't use any explicit or harmful content when running experiments and demos on our tool.

            ---

            ## Evaluation metrics

            We chose to aim our evaluations at specific types of protection,

            - **Instruction-based editing** - We test our model against InstructPix2Pix-based editors, with prompts that change the human faces.
            - **Inpainting editings** - We test our model against Stable Diffusion, with specific instruction prompts that edits the background of any image,
            without editing the face.

            ---

            """
        )
        gr.HTML(
            """
            <div style="margin-top:8px;">
              <a onclick="window.showPage('main')"
                 style="cursor:pointer; display:inline-block; padding:10px 22px;
                        background:#2563eb; color:white; border-radius:8px;
                        font-weight:600; font-size:14px; text-decoration:none;">
                &larr; Back to the Tool
              </a>
            </div>
            """
        )

    demo.load(fn=None, js=ALL_JS)

def launch() -> None:
    cleanup_old_outputs()
    open_browser = os.environ.get("IMAGESHIELD_OPEN_BROWSER", "1") != "0"
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=open_browser,
        share=False,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=CUSTOM_CSS,
        allowed_paths=[str(OUTPUT_DIR)],
    )


if __name__ == "__main__":
    launch()
