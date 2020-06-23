import argparse
import json
from model import FlowerModel
from PIL import Image
from torchvision import  transforms

def process_image(image):
    prc_img = Image.open(image)
   
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    output = img_transform(prc_img)
    
    return output

def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", action="store")
    parser.add_argument("checkpoint_path", action="store")
    parser.add_argument("--top_k", action="store", default=1, type=int)
    parser.add_argument("--category_names", action="store",
                        default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", default=False)
    return parser.parse_args()


def load_name():
    cat_to_name=''
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

if __name__ == "__main__":
    args = cli_options()

    cat_to_name = load_name()

    image = process_image(args.image_path)

    fm = FlowerModel.loading_checkpoint(args.checkpoint_path)

    print(f"Predict flower class for image {args.image_path} ..")

    ps, class = fm.predict(image, args.top_k)
    for i, c in enumerate(top_class):
        print(f"Prediction {i+1}: "
              f"{cat_to_name[c]} .. "
              f"({100.0 * top_ps[i]:.3f}%)")