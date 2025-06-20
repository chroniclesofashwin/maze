# AI Video Generator (Text + Image/Video → Video)
# Open-Source CLI Prototype using AnimateDiff + Stable Diffusion

import argparse
import os
from PIL import Image
from torchvision import transforms
import torch

# Import AnimateDiff, Stable Diffusion pipelines (assumed installed)
# You must have pre-trained models downloaded or linked.
from diffusers import StableDiffusionPipeline
from animatediff.pipeline import AnimateDiffPipeline


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def generate_animation(prompt, image_path, output_path, steps=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[+] Loading Stable Diffusion Pipeline...")
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print("[+] Loading AnimateDiff Pipeline...")
    anim_pipeline = AnimateDiffPipeline.from_pretrained(
        "animatediff/animatediff-text-to-video",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Prepare inputs
    init_image = load_image(image_path).to(device)

    print("[+] Generating animation...")
    output = anim_pipeline(
        prompt=prompt,
        init_image=init_image,
        num_frames=16,
        num_inference_steps=steps,
        guidance_scale=7.5
    )

    print("[+] Saving output video...")
    output["video"].save(output_path)
    print(f"[✓] Video saved at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate AI Video from prompt + image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video")
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion steps")
    args = parser.parse_args()

    generate_animation(
        prompt=args.prompt,
        image_path=args.image,
        output_path=args.output,
        steps=args.steps
    )


if __name__ == "__main__":
    main()
