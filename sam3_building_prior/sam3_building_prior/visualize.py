from PIL import Image
import numpy as np


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color=(255,0,0), alpha=0.4):
    """Overlay a binary mask on top of an RGB image for quick inspection."""
    if image.mode != "RGBA":
        bg = image.convert("RGBA")
    else:
        bg = image.copy()
    overlay = Image.new("RGBA", bg.size, (0,0,0,0))
    mask_img = Image.fromarray((mask.astype('uint8')*255)).convert("L")
    overlay.paste(Image.new("RGBA", bg.size, color + (int(255*alpha),)), mask=mask_img)
    out = Image.alpha_composite(bg, overlay)
    return out.convert("RGB")
