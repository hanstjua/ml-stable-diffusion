from python_coreml_stable_diffusion import unet
from python_coreml_stable_diffusion.torch2coreml import convert_vae_decoder, convert_vae_encoder, convert_unet, convert_text_encoder, convert_safety_checker

from diffusers import StableDiffusionPipeline

class Args:
    pass

def convert(model: str, width: int, height: int, output_dir: str):
    args = Args()
    args.convert_text_encoder = True
    args.convert_vae_decoder = True
    args.convert_vae_encoder = True
    args.convert_unet = True
    args.convert_safety_checker = True
    args.convert_controlnet = ''
    args.model_version = model
    args.compute_unit = 'ALL'
    args.latent_h = height
    args.latent_w = width
    args.attention_implementation = 'SPLIT_EINSUM'
    args.o = output_dir
    args.check_output_correctness = False
    
    # Instantiate diffusers pipe as reference
    pipe = StableDiffusionPipeline.from_ckpt(args.model_version)

    # Register the selected attention implementation globally
    unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations[
        args.attention_implementation]
    print(
        f"Attention implementation in effect: {unet.ATTENTION_IMPLEMENTATION_IN_EFFECT}"
    )

    # Convert models
    if args.convert_vae_decoder:
        convert_vae_decoder(pipe, args)

    if args.convert_vae_encoder:
        convert_vae_encoder(pipe, args)
        
    if args.convert_unet:
        convert_unet(pipe, args)

    if args.convert_text_encoder:
        convert_text_encoder(pipe, args)

    if args.convert_safety_checker:
        convert_safety_checker(pipe, args)

