import gradio as gr
import torch
from optimum.quanto import freeze, quantize, qint2, qint4, qint8, qfloat8
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
import time
import argparse
import random

UNET_QTYPES = {
    "fp8": qfloat8,
    "int8": qint8,
    "int4": qint4,
    "int2": qint2,
    "none": None,
}

pipe = None

def create_pipe(model: str, dtype = "bfloat16", offload = True, weight = "int8", lora_repo_id = None, lora_weights = None):
    if model == "flux-schnell":
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
    elif model == "flux-dev":
        bfl_repo = "black-forest-labs/FLUX.1-dev"
    else:
        raise ValueError(f"Invalid model name: {model}")

    start_time = time.time()
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    pipe = FluxPipeline.from_pretrained(
        bfl_repo, torch_dtype=torch_dtype
    )

    if lora_repo_id is not None:
        pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict=lora_repo_id,
            weight_name=lora_weights
        )

    if weight != "none":
        quantize(pipe.transformer, weights=UNET_QTYPES[weight], exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
        freeze(pipe.transformer)
        quantize(pipe.text_encoder_2, weights=UNET_QTYPES[weight])
        freeze(pipe.text_encoder_2)

    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"パイプラインの作成時間: {execution_time:.2f} 秒")
    return pipe

def generate_image(width, height, num_steps, guidance, seed, prompt):
    start_time = time.time()
    
    if seed == "-1" or seed == "":
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(seed)
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance,
        output_type="pil",
        num_inference_steps=num_steps,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"画像生成の所要時間: {execution_time:.2f} 秒")
    print(f"使用されたシード値: {seed}")
    return image, seed

def create_demo(model: str, weight: str):
    is_schnell = model == "flux-schnell"

    with gr.Blocks() as demo:
        gr.Markdown(f"# Flux.1 画像生成デモ（Diffusers版） - モデル: {model} - 量子化精度: {weight}")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="3D animation style graphics reminiscent of Pixar of a magical girl in a pink costume. She is holding a magic wand. Beside her is a white cat. Seven-colored lights, petals and butterflies are dancing around them.")
               
                with gr.Accordion("詳細設定", open=False):
                    width = gr.Slider(128, 8192, 1360, step=16, label="幅")
                    height = gr.Slider(128, 8192, 768, step=16, label="高さ")
                    num_steps = gr.Slider(1, 50, 4 if is_schnell else 20, step=1, label="ステップ数")
                    guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="ガイダンス", interactive=not is_schnell)
                    seed = gr.Textbox(-1, label="シード値 (-1 ：ランダム)")
                
                generate_btn = gr.Button("画像生成")
            
            with gr.Column():
                output_image = gr.Image(label="生成画像", show_download_button=True)
                generation_time = gr.Markdown(label="生成時間")
                seed_used = gr.Markdown(label="使用されたシード値")

        def generate_and_display(width, height, num_steps, guidance, seed, prompt):
            start_time = time.time()
            image, seed = generate_image(width, height, num_steps, guidance, seed, prompt)
            end_time = time.time()
            execution_time = end_time - start_time
            return image, f"使用されたシード値: {seed}", f"画像生成の所要時間: {execution_time:.2f} 秒"
        
        generate_btn.click(
            fn=generate_and_display,
            inputs=[width, height, num_steps, guidance, seed, prompt],
            outputs=[output_image, seed_used, generation_time], queue=True
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux.1 with Diffusers demo app")
    parser.add_argument("--model", type=str, default="flux-schnell", choices=["flux-schnell", "flux-dev"], help="Model name(default: flux-schnell)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Data type(default: bfloat16)")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--weight", type=str, default="int8", choices=["int2", "int4", "int8", "fp8", "none"], help="quantization precision(default: int8)")
    parser.add_argument("--lora_repo_id", type=str, default=None, help="LoRA repository ID or path")
    parser.add_argument("--lora_weights", type=str, default=None, help="LoRA weights name")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--inbrowser", action="store_true", help="Launch the demo in the browser")
 
    args = parser.parse_args()

    pipe = create_pipe(model=args.model, dtype=args.dtype, offload=args.offload, weight=args.weight, lora_repo_id =args.lora_repo_id, lora_weights=args.lora_weights)
    demo = create_demo(model=args.model, weight=args.weight)
    demo.launch(share=args.share, inbrowser=args.inbrowser)