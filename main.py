import os 
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
from shuffle_batchnorm import ShuffleBatchNorm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--epochs', default=10, type=int)


class SimpleCNN(nn.Module):
    def __init__(self, num_class=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8,num_class)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def main():
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0) 

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    if ',' in args.gpu: args.num_gpu = len(args.gpu.split(','))
    else: args.num_gpu = 1
    args.distributed = args.num_gpu > 1

    if args.distributed:
        print('=> Spawning %d distributed workers' % args.num_gpu)
        mp.spawn(main_worker, nprocs=args.num_gpu, args=(args,))
    else:
        main_worker(gpu_id=0, args=args)


def main_worker(gpu_id, args):
    args.gpu_id = gpu_id
    prefix = '[%d]' % gpu_id

    if args.distributed:
        args.rank = gpu_id
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group(backend='nccl', world_size=args.num_gpu, rank=args.rank)
        torch.manual_seed(666) # random seed: all gpus init same model parameter

    model = SimpleCNN(num_class=2)
    print(prefix+'model created')

    # shuffle BN
    if args.distributed and args.shuffle:
        model = ShuffleBatchNorm.convert_shuffle_batchnorm(model)
        print(prefix+'convert to shuffleBN')

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu_id)

    if args.distributed:
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)
        model = nn.parallel.DistributedDataParallel(model, 
            device_ids=[args.gpu_id], broadcast_buffers=False)
    else:
        torch.cuda.set_device(args.gpu_id)
        model = model.cuda(args.gpu_id)

    if args.distributed: torch.manual_seed(gpu_id) # random seed: all gpus generate different data

    # randomly generate some data
    B = 4
    H,W = 8,8
    model.train()
    for i in range(args.epochs):
        data = torch.FloatTensor(B,3,H,W).cuda(args.gpu_id)
        nn.init.normal_(data)
        target = torch.from_numpy(np.random.choice(range(2), B)).to(data.device).long()
        logit = model(data)
        loss = criterion(logit, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[%d/%d] Loss %.4f' % (i,args.epochs,loss.item()))
        print('================================================')


if __name__ == '__main__':
    main()
