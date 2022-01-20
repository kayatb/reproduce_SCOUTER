""" Code from scouter/test.py altered to calculate the average attention ratio/area size for the complete validation
set in ImageNet. """

from __future__ import print_function
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os, os.path

from sloter.slot_model import SlotModel
from train import get_args_parser
from dataset.ConText import ConText, MakeListImage

def calc_area_size(args, model, device, image, label):
    model.to(device)
    model.eval()
    image = image.to(device, dtype=torch.float32)
    _ = model(torch.unsqueeze(image, dim=0))

    model.train()

    # TODO: least similar class for negative scouter
    if args.loss_status > 0:
        id = label
    else:
        id = label + 1

    slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png'), dtype=np.uint8)
    slot_image_size = slot_image.shape
    attention_ratio = float(slot_image.sum()) / float(slot_image_size[0]*slot_image_size[1]*255)
    
    return attention_ratio


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    parser.add_argument("--csv", default="data/imagenet/LOC_val_solution.csv", type=str, help="Location of the CSV file that contains the bounding boxes")
    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    # Directory to save images during model forward pass.
    os.makedirs('sloter/vis', exist_ok=True)

    model_name = f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"\
                + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"\
                + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}" + 'checkpoint.pth'

    args.use_pre = False

    device = torch.device(args.device)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        ])

    # Retrieve the data. We only need to evaluate the validation set.
    _, val = MakeListImage(args).get_data()
    dataset_val = ConText(val, transform=transform)
    # Use batch size = 1 to handle a single image at a time.
    data_loader_val = torch.utils.data.DataLoader(dataset_val, 1, shuffle=False, num_workers=1, pin_memory=True)

    # Load the model from checkpoint.
    model = SlotModel(args)
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    total_area_size = 0
    num_points = 0
    # Process each image.
    for data in data_loader_val:
        image = data["image"][0]
        label = data["label"][0].item()

        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform2 = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        image = transform2(image)

        total_area_size += calc_area_size(args, model, device, image, label)
        num_points += 1

        if num_points%1000 == 0:
            print(f"Processed {num_points} images!")

    print(f"Average area size is: {total_area_size / num_points}")


if __name__ == '__main__':
    main()
