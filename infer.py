import torch
from src.utils.utils import validate, create_dataloader
from src.models import create_model
import argparse

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch STAM Kinetics Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='stam_16')
parser.add_argument('--num_classes', type=int, default=400)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--frames_per_clip', type=int, default=16)
parser.add_argument('--frame_rate', type=float, default=1.6)
parser.add_argument('--step_between_clips', type=int, default=1000)




def main():
    # parsing args
    args = parser.parse_args()

    # setup model
    print('creating model...')
    model = create_model(args).cuda()
    state = torch.load(args.model_path, map_location='cpu')['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    print('done\n')

    # setup data loader
    print('creating data loader...')
    val_loader = create_dataloader(args)
    print('done\n')

    # actual validation process
    print('doing validation...')
    prec1_f, prec5_f = validate(model, val_loader)
    print("final top-1 validation accuracy: {:.2f}".format(prec1_f.avg))


if __name__ == '__main__':
    main()