import os
import argparse
from glob import glob

import torch

from utils.utils import load_model
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageDraw
import os

# 创建网格保存图片
def save_images_in_grid(images, save_path, grid_size=(4, 4), img_size=(256, 256), names=None):
    grid_width, grid_height = grid_size
    grid_img = Image.new("RGB", (img_size[0] * grid_width, img_size[1] * grid_height))
    
    for i, img in enumerate(images):
        if names:
            img_name = names[i]
            img = img.resize(img_size)
            x = (i % grid_width) * img_size[0]
            y = (i // grid_width) * img_size[1]
            grid_img.paste(img, (x, y))
            
            # 在图像上写上名称
            draw = ImageDraw.Draw(grid_img)
            draw.text((x + 10, y + 10), img_name, fill=(255, 255, 255))
    
    # 获取当前的日期时间并格式化为字符串（年-月-日_时-分-秒）
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    savepath=os.path.join(save_path,current_time)
    # 如果路径不存在，则创建路径
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # 定义文件名
    file_name = "grid_image.png"
    savepath = os.path.join(savepath, file_name)

    grid_img.save(savepath, format="PNG")
    print(f"网格图片已保存到 {savepath}")

# 执行推理并保存到网格
img_results = []
img_names = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='sample/cat1.png', help="img file path")
    parser.add_argument('--prompt', type=str, help="source(reference) prompt")
    parser.add_argument('--trg_prompt', type=str, nargs='+', help="target prompt")
    parser.add_argument('--num_inference_steps', type=int, default=200, help="inference steps")
    parser.add_argument('--save_path', type=str, default='results', help="save directory")
    parser.add_argument('--w_cut', type=float, default=3.0, help="weight coefficient for cut loss term")
    parser.add_argument('--w_dds', type=float, default=1.0, help="weight coefficient for dds loss term")
    parser.add_argument('--patch_size', type=int, nargs='+', default=[1,2], help="size of patches")
    parser.add_argument('--n_patches', type=int, default=256, help="number of patches")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--cuda', type=int, default=0, help="gpu device id")
    parser.add_argument('--v5', action='store_true', default=False)
    parser.add_argument("--torch_dtype", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="dtype for less vram memory")
    args = parser.parse_args()

    # Prepare model
    device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_model(args)

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    if os.path.isdir(args.img_path):
        img_files = sorted(glob(os.path.join(args.img_path, '*.jpg')))
    else:
        img_files = [args.img_path]

    # 初始化一个列表来存储所有的图片和标签
    images_inversion_list = []
        
    # Inference
    for img_file in img_files:
        print(img_file)
        
        result = stable(
            img_path=img_file,
            prompt=args.prompt,
            trg_prompt=args.trg_prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            n_patches=args.n_patches,
            patch_size=args.patch_size,
            save_path=args.save_path,
        )

        img_results.append(result)
        img_names.append(os.path.basename(img_file))  # 使用文件名作为图片名称
    
    # 将所有生成的图片保存到网格
    save_images_in_grid(img_results, save_path=args.save_path, grid_size=(4, 4), img_size=(256, 256), names=img_names)      

if __name__ == '__main__':    
    main()
