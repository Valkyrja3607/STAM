import torch
#from src.utils.utils import validate, create_dataloader
import datetime
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torchvision
import numpy as np
import copy
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from src.models import create_model
import argparse
from config import Config
from ucf101 import UCF101

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch STAM Kinetics Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='stam_16')
parser.add_argument('--num_classes', type=int, default=101)
parser.add_argument('--input_size', type=int, default=112)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--frames_per_clip', type=int, default=16)
parser.add_argument('--frame_rate', type=float, default=1.6)
parser.add_argument('--step_between_clips', type=int, default=1000)


def video_collate(batch):
  is_np = isinstance(batch[0][0][0], np.ndarray)
  T = len(batch[0][0])  # number of frames
  targets = torch.tensor([b[2] for b in batch])
  if len(batch[0]) == 3:
    extra_data = [b[1] for b in batch]
  else:
    extra_data = []
  batch_size = len(batch)
  if is_np:
    dims = (batch[0][0][0].shape[2], batch[0][0][0].shape[0], batch[0][0][0].shape[1])
    tensor_uint8_CHW = torch.empty((T * batch_size, *dims), dtype=torch.uint8)
    for i in range(batch_size):
      for t in range(T):
        tensor_uint8_CHW[i * T + t] = \
          torch.from_numpy(batch[i][0][t]).permute(2, 0, 1)
    return tensor_uint8_CHW, targets

  else:
    dims = (batch[0][0][0].shape[0], batch[0][0][0].shape[1], batch[0][0][0].shape[2])
    tensor_float_CHW = torch.empty((T * batch_size, *dims), dtype=torch.float)
    for i in range(batch_size):
      for t in range(T):
        tensor_float_CHW[i * T + t] = batch[i][0][t]
    return tensor_float_CHW, targets


def main():
    # parsing args
    args = parser.parse_args()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((112,112)),
        torchvision.transforms.ToTensor(),
    ])

    def load_video(video_path: str):
        image_size = 112
        video = torchvision.io.read_video(video_path, pts_unit="sec")[0].float()

        # [t, n, h, w]
        video = video.permute(0, 3, 1, 2)
        t_size = 16
        #new_video = torch.zeros((min(video.shape[0], t_size), video.shape[1], image_size, image_size))
        new_video = torch.zeros((t_size, video.shape[1], image_size, image_size))
        for t in range(min(video.shape[0], t_size)):
            if video.shape[0] < t_size:
                frame = transforms(video[t])
                new_video[t] = (frame - 127.5) / 127.5
            else:
                t_ = int(video.shape[0]/t_size*t)
                frame = transforms(video[t_])
                new_video[t] = (frame - 127.5) / 127.5

        return new_video
        
    def validate(model, val_loader):
        device = 'cuda'
        model.eval()
        total, correct = 0, 0
        for j, (video, label) in enumerate(val_loader):
            #batch = torch.tensor(batch)
            #video, label = batch[:,0], batch[:,1]
            video, label = video.view(-1,3,112,112), label.view(-1)
            video = video.to(device)
            #label = label.cuda()
            with torch.no_grad():
                pred = model(video)
                pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            correct += accuracy_score(label, pred, normalize=False)
            total += 16#batch_size

        return correct / total * 100

    def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, max_epochs):
        device = 'cuda'
        model.train()
        loss_fn = loss_fn.to(device)
        optimizer.zero_grad()
        #loss = torch.as_tensor(0.0).cuda()
        best_acc = 0
        best_model = copy.deepcopy(model)
        for i in range(max_epochs):
            for j, (video, label) in enumerate(train_loader):
                #batch = torch.tensor(batch)
                #video, label = batch[:,0], batch[:,1]
                video, label = video.view(-1,3,112,112), label.view(-1)
                video = video.to(device)
                label = label.to(device)
                pred = model(video)
                pred = F.softmax(pred, dim=1)
                loss = loss_fn(pred, label)
                #torch.cuda.empty_cache()
                #loss /= len(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if j%100==0:
                    print(f"epoch:{i}, {j}, loss:{loss.item()}")
                    wandb.log({"loss": loss.item()})
                if j%1000==0:
                    acc = validate(model, val_loader)
                    wandb.log({"acc": acc})
                    if acc > best_acc:
                        best_acc = acc
                        best_model = copy.deepcopy(model)
                    print("best_acc:",best_acc)
        
        return best_model


    wandb.init(project="stam", name=f"stam-{datetime.datetime.now()}")

    # k times cross validation
    for i in range(1,4):
        cfg = Config()
        device = "cuda"
        train_dataset = UCF101(
                "./dataset/UCF101",
                [
                    f"./dataset/ucfTrainTestlist/trainlist0{i}.txt",
                    f"./dataset/ucfTrainTestlist/trainlist0{((i+1)%3)+1}.txt",
                ],
            )
        testset = UCF101(
                "./dataset/UCF101",
                [f"./dataset/ucfTrainTestlist/trainlist0{((i+2)%3)+1}.txt"],
            )
        
        t = len(testset)
        t, v = int(t*0.9), t-int(t*0.9) 
        test_dataset, val_dataset = torch.utils.data.random_split(testset, [t, v])
        train_loader = DataLoader(
            train_dataset,
            1,
            shuffle=True,
            #collate_fn=video_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            1,
            shuffle=True,
            #collate_fn=video_collate,
        )
        test_loader = DataLoader(
            test_dataset,
            1,
            shuffle=False,
            #collate_fn=video_collate,  
        )

        # setup model
        print('creating model...')
        model = create_model(args).to(device)
        #state = torch.load(args.model_path, map_location='cpu')['model']
        #model.load_state_dict(state, strict=False)
        #model.eval()
        print('done\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (512 ** -0.5) * min((x+1) ** (-0.5), (x+1) * cfg.warmup_steps ** (-1.5)),
        )
        loss_fn = nn.CrossEntropyLoss()

        print(sum(p.numel() for p in model.parameters()))

        model = train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, max_epochs=cfg.epochs)
        torch.save(model, f"./checkpoints/ckpt-{i}.pt")
        acc = validate(model, test_loader)
        print(f"{i}acc:{acc}")
        wandb.log({"test_acc": acc})



if __name__ == '__main__':
    main()