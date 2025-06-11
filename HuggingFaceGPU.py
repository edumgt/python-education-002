from diffusers import StableDiffusionPipeline
import torch

# GPU로 파이프라인 로딩
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# 프롬프트로 이미지 생성
prompt = "Many Fruit, Natural Sun Light, 4K"
image = pipe(prompt, guidance_scale=7.5).images[0]

# 저장
image.save("diffusers_gpu_output.png")
print("✅ Diffusers GPU 이미지 생성 완료")
