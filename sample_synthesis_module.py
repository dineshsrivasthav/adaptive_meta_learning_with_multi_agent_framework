from segment_anything import build_sam, SamPredictor
import torch
import numpy as np
from PIL import Image
import os
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline
from multiagents import run_multiagent_workflow, extract_prompts_from_result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize models (lazy loading)
_groundingdino_model = None
_sam_predictor = None
_sd_pipe = None


def initialize_models():
    """Initialize all models needed for sample synthesis."""
    global _groundingdino_model, _sam_predictor, _sd_pipe
    
    if _groundingdino_model is None:
        print("Loading GroundingDINO model...")
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        _groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
    
    if _sam_predictor is None:
        print("Loading SAM model...")
        sam_checkpoint = 'sam_vit_h_4b8939.pth'
        _sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    
    if _sd_pipe is None:
        print("Loading Stable Diffusion model...")
        _sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting"
        ).to(device)
    
    return _groundingdino_model, _sam_predictor, _sd_pipe


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    """Load GroundingDINO model from HuggingFace."""
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    
    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} => {log}")
    _ = model.eval()
    return model


def detect_objects(image, text_prompt, model, box_threshold=0.3, text_threshold=0.25):
    """Detect objects in image using GroundingDINO."""
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        device=device,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    return boxes


def segment_objects(image, sam_model, boxes):
    """Segment objects using SAM."""
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()


def generate_image(image, mask, prompt, negative_prompt, pipe, seed=32):
    """Generate image using Stable Diffusion inpainting."""
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))
    
    generator = torch.Generator(device).manual_seed(seed)
    
    result = pipe(
        image=in_image, 
        mask_image=in_mask, 
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        generator=generator
    )
    result = result.images[0]
    
    return result.resize((w, h))


def inpaint(image_path: str, text_prompt: str, mask_type: str, 
           positive_prompt: str, negative_prompt: str,
           verbose: bool = False) -> Image.Image:
    """
    Generate a deepfake sample using inpainting.
    
    Args:
        image_path: Path to the base image
        text_prompt: Text prompt for object detection
        mask_type: 'normal' or 'inverted'
        positive_prompt: Prompt for generating the base image
        negative_prompt: Prompt describing the deepfake modifications
        verbose: Whether to print intermediate steps
    
    Returns:
        Generated image
    """
    groundingdino_model, sam_predictor, sd_pipe = initialize_models()
    
    image_source, image = load_image(image_path)
    if verbose:
        print('Original image loaded')
    
    # Detect objects
    detected_boxes = detect_objects(image, text_prompt, groundingdino_model)
    if verbose:
        print(f'Detected {len(detected_boxes)} objects')
    
    # Segment objects
    segmented_frame_masks = segment_objects(image_source, sam_predictor, boxes=detected_boxes)
    if verbose:
        print('Objects segmented')
    
    # Create mask
    mask = segmented_frame_masks[0][0].cpu().numpy()
    inverted_mask = ((1 - mask) * 255).astype(np.uint8)
    
    image_source_pil = Image.fromarray(image_source)
    image_mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    inverted_image_mask_pil = Image.fromarray(inverted_mask)
    
    # Select mask type
    if mask_type == 'inverted':
        mask_pil = inverted_image_mask_pil
    else:
        mask_pil = image_mask_pil
    
    # Generate image
    generated_image = generate_image(
        image=image_source_pil, 
        mask=mask_pil, 
        prompt=positive_prompt, 
        negative_prompt=negative_prompt, 
        pipe=sd_pipe, 
        seed=32
    )
    
    return generated_image


def load_prompts_from_multiagents(output_dir: str = "./outputs") -> Dict[str, List[str]]:
    """
    Load prompts generated by the multi-agent workflow.
    
    Args:
        output_dir: Directory where multiagent outputs are stored
    
    Returns:
        Dictionary with 'positive_prompts' and 'negative_prompts'
    """
    prompts_file = os.path.join(output_dir, "promptsm.md")
    
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(
            f"Prompts file not found at {prompts_file}. "
            "Please run the multi-agent workflow first."
        )
    
    prompts = {'positive_prompts': [], 'negative_prompts': []}
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse prompts from markdown
    # Look for patterns like "Positive Prompt:" or "Prompt 1:" etc.
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        if 'positive' in line_lower or 'prompt 1' in line_lower or 'base prompt' in line_lower:
            current_section = 'positive'
        elif 'negative' in line_lower or 'prompt 2' in line_lower or 'modification prompt' in line_lower:
            current_section = 'negative'
        elif line.strip() and not line.strip().startswith('#') and current_section:
            # Extract the actual prompt text
            prompt_text = line.strip()
            if len(prompt_text) > 10:  # Filter out very short lines
                prompts[f'{current_section}_prompts'].append(prompt_text)
    
    # If parsing failed, try to extract from result string directly
    if not prompts['positive_prompts'] and not prompts['negative_prompts']:
        # Fallback: try to extract JSON or structured format
        import re
        # Look for JSON-like structures
        json_match = re.search(r'\{.*"positive.*"negative.*\}', content, re.DOTALL)
        if json_match:
            try:
                prompts = json.loads(json_match.group())
            except:
                pass
    
    return prompts


def generate_few_shot_samples(
    prompts: Dict[str, List[str]],
    base_image_dir: str,
    output_dir: str = "./few_shot_samples",
    text_prompt: str = "person face",
    mask_type: str = "normal",
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Generate few-shot samples from prompts and save them to a directory.
    
    Args:
        prompts: Dictionary with 'positive_prompts' and 'negative_prompts'
        base_image_dir: Directory containing base images to use
        output_dir: Directory to save generated samples
        text_prompt: Text prompt for object detection
        mask_type: 'normal' or 'inverted'
        max_samples: Maximum number of samples to generate (None for all)
    
    Returns:
        List of paths to generated sample images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    positive_prompts = prompts.get('positive_prompts', [])
    negative_prompts = prompts.get('negative_prompts', [])
    
    if not positive_prompts or not negative_prompts:
        raise ValueError("No prompts found. Please ensure prompts are properly formatted.")
    
    # Match positive and negative prompts
    num_pairs = min(len(positive_prompts), len(negative_prompts))
    if max_samples:
        num_pairs = min(num_pairs, max_samples)
    
    # Get base images
    if os.path.isdir(base_image_dir):
        base_images = [os.path.join(base_image_dir, f) for f in os.listdir(base_image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        base_images = [base_image_dir] if os.path.isfile(base_image_dir) else []
    
    if not base_images:
        raise ValueError(f"No base images found in {base_image_dir}")
    
    generated_samples = []
    metadata = []
    
    print(f"Generating {num_pairs} few-shot samples...")
    
    for i in range(num_pairs):
        base_image_path = base_images[i % len(base_images)]
        positive_prompt = positive_prompts[i]
        negative_prompt = negative_prompts[i % len(negative_prompts)]
        
        try:
            # Generate sample
            generated_image = inpaint(
                image_path=base_image_path,
                text_prompt=text_prompt,
                mask_type=mask_type,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                verbose=False
            )
            
            # Save sample
            sample_filename = f"sample_{i+1:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            sample_path = os.path.join(output_dir, sample_filename)
            generated_image.save(sample_path)
            generated_samples.append(sample_path)
            
            # Save metadata
            metadata.append({
                'sample_path': sample_path,
                'base_image': base_image_path,
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt,
                'text_prompt': text_prompt,
                'mask_type': mask_type,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Generated sample {i+1}/{num_pairs}: {sample_filename}")
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(generated_samples)} samples in {output_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    return generated_samples


def run_sample_synthesis_pipeline(
    user_query: str,
    base_image_dir: str,
    output_dir: str = "./few_shot_samples",
    multiagent_output_dir: str = "./outputs",
    text_prompt: str = "person face",
    mask_type: str = "normal",
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Complete pipeline: Run multi-agent workflow -> Generate few-shot samples.
    
    Args:
        user_query: Query for the multi-agent workflow
        base_image_dir: Directory containing base images
        output_dir: Directory to save generated samples
        multiagent_output_dir: Directory for multi-agent outputs
        text_prompt: Text prompt for object detection
        mask_type: 'normal' or 'inverted'
        max_samples: Maximum number of samples to generate
    
    Returns:
        List of paths to generated sample images
    """
    print("="*60)
    print("Step 1: Running multi-agent workflow to generate prompts...")
    print("="*60)
    
    # Run multi-agent workflow
    prompts = run_multiagent_workflow(user_query, multiagent_output_dir)
    
    # If prompts are empty, try to load from file
    if not prompts.get('positive_prompts') and not prompts.get('negative_prompts'):
        print("Loading prompts from output file...")
        prompts = load_prompts_from_multiagents(multiagent_output_dir)
    
    print(f"\nFound {len(prompts.get('positive_prompts', []))} positive prompts")
    print(f"Found {len(prompts.get('negative_prompts', []))} negative prompts")
    
    print("\n" + "="*60)
    print("Step 2: Generating few-shot samples...")
    print("="*60)
    
    # Generate samples
    generated_samples = generate_few_shot_samples(
        prompts=prompts,
        base_image_dir=base_image_dir,
        output_dir=output_dir,
        text_prompt=text_prompt,
        mask_type=mask_type,
        max_samples=max_samples
    )
    
    return generated_samples


# Main execution for standalone use
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate few-shot deepfake samples")
    parser.add_argument('--user_query', type=str, 
                       default='Write 15 Few-shot Prompts to create novel Image deepfakes',
                       help='Query for multi-agent workflow')
    parser.add_argument('--base_image_dir', type=str, required=True,
                       help='Directory containing base images')
    parser.add_argument('--output_dir', type=str, default='./few_shot_samples',
                       help='Directory to save generated samples')
    parser.add_argument('--text_prompt', type=str, default='person face',
                       help='Text prompt for object detection')
    parser.add_argument('--mask_type', type=str, default='normal',
                       choices=['normal', 'inverted'],
                       help='Mask type for inpainting')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to generate')
    
    args = parser.parse_args()
    
    samples = run_sample_synthesis_pipeline(
        user_query=args.user_query,
        base_image_dir=args.base_image_dir,
        output_dir=args.output_dir,
        text_prompt=args.text_prompt,
        mask_type=args.mask_type,
        max_samples=args.max_samples
    )
    
    print(f"\nSample synthesis completed... Generated {len(samples)} samples.")
