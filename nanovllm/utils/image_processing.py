import math
import torch
from PIL import Image
import torchvision.transforms.functional as TF


IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)

def smart_resize(height: int, width: int, factor: int = 32,
                 min_pixels: int = 3136, max_pixels: int = 1003520) -> tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(f"Image too small: {height}x{width}, min edge = {factor}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        scale = math.sqrt(max_pixels / (h_bar * w_bar))
        h_bar = max(factor, math.floor(h_bar * scale / factor) * factor)
        w_bar = max(factor, math.floor(w_bar * scale / factor) * factor)

    if h_bar * w_bar < min_pixels:
        scale = math.sqrt(min_pixels / (h_bar * w_bar))
        h_bar = max(factor, math.ceil(h_bar * scale / factor) * factor)
        w_bar = max(factor, math.ceil(w_bar * scale / factor) * factor)

    return h_bar, w_bar

def process_image(
    image: Image.Image,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 1003520,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = image.convert("RGB")
    w, h = image.size
    factor = patch_size * merge_size

    new_h, new_w = smart_resize(h, w, factor, min_pixels, max_pixels)
    image = image.resize((new_w, new_h), Image.BICUBIC)

    pixels = TF.to_tensor(image)
    pixels = TF.normalize(pixels, IMAGE_MEAN, IMAGE_STD)

    pixels = pixels.unsqueeze(0)
    if pixels.shape[0] < temporal_patch_size:
        pad = pixels.repeat(temporal_patch_size, 1, 1, 1)[:temporal_patch_size]
        pixels = pad

    T, C, H, W = pixels.shape
    grid_t = T // temporal_patch_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    patches = pixels.view(
        grid_t, temporal_patch_size,
        C,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    pixel_values = patches.reshape(-1, C * temporal_patch_size * patch_size * patch_size)

    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)

    return pixel_values, image_grid_thw

def process_messages(
    msg: dict,
    tokenizer,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_token_id: int = 248056,
    vision_start_id: int = 248053,
    vision_end_id: int = 248054,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
    pixel_values_list = []
    grid_thw_list = []

    text_parts = []
        
    role = msg["role"]
    content = msg["content"]
    if isinstance(content, str):
        text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    elif isinstance(content, list):
        parts_text = []
        for part in content:
            if part["type"] == "image":
                img = part["image"]
                if isinstance(img, str):
                    img = Image.open(img)
                pv, gt = process_image(img, patch_size, temporal_patch_size, merge_size)
                pixel_values_list.append(pv)
                grid_thw_list.append(gt)
                # Compute number of merged tokens
                t, h, w = gt[0].tolist()    # 1, 32, 64
                num_tokens = t * (h // merge_size) * (w // merge_size)  # 1*32/2*64/2=512
                placeholder = "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"
                parts_text.append(placeholder)
            elif part["type"] == "text":
                parts_text.append(part["text"])
        text_parts.append(f"<|im_start|>{role}\n{''.join(parts_text)}<|im_end|>\n")

    full_text = "".join(text_parts) + "<|im_start|>assistant\n"
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if pixel_values_list:
        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.cat(grid_thw_list, dim=0)
    else:
        pixel_values = None
        image_grid_thw = None

    return token_ids, pixel_values, image_grid_thw
