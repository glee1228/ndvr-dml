from tqdm import tqdm
import numpy as np
import argparse
import warnings
import os

from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torch_optimizer
from script_dh.mnist import MNIST_t
from ndisDB_script import models
import datasets
from utils import *
import torch.nn.functional as F
from torch.autograd import Variable

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = None
writer = None


@torch.no_grad()
def eval_vcdb_frame_rank(net, loader, vcdb, epoch, chunk=1000, margin=1):
    loader.dataset.l = [f for v in vcdb.core_videos for f in vcdb.get_frames(v)]
    features = extract_feature(net, loader, progress=True, single=True)
    length = [vcdb.get_framecount(v) for v in vcdb.core_videos]
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    ranks = vcdb_frame_retrieval(vcdb, features, index, chunk, margin, progress=True)
    avg_rank, avg_dist, avg_rank_per_query, avg_dist_per_query = ranks
    logger.info(f'[Epoch {epoch}] '
                f'Rank/Frame: {avg_rank:4f}, Rank/Seg :{avg_rank_per_query:4f}, '
                f'Distance/Frame: {avg_dist:4f}, Distance/Seg: {avg_dist_per_query:4f}')

    writer.add_scalar('test/rank_per_frame', avg_rank, epoch)
    writer.add_scalar('test/distance_per_frame', avg_dist, epoch)
    writer.add_scalar('test/rank_per_query', avg_rank_per_query, epoch)
    writer.add_scalar('test/distance_per_query', avg_dist_per_query, epoch)

    del features


def train(net, loader, optimizer, criterion, scheduler, epoch):
    losses = AverageMeter()
    lr = AverageMeter()
    net.train()
    bar = tqdm(loader, ncols=150)
    for batch_idx, (data1, data2, data3) in enumerate(loader):
        optimizer.zero_grad()
        # print()
        # print(path[0][0].split('/')[-3:],path[1][0].split('/')[-3:],path[2][0].split('/')[-3:])
        _output1, _output2, _output3 = net(data1.float().cuda(), data2.float().cuda(), data3.float().cuda())
        dist_a = F.pairwise_distance(_output1, _output2, 2)
        dist_b = F.pairwise_distance(_output1, _output3, 2)
        target = torch.FloatTensor(dist_a.size()).fill_(1).cuda()
        target = Variable(target)
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = _output1.norm(2) + _output2.norm(2) + _output3.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        # loss = criterion(_output1, _output2, _output3)
        # import pdb;pdb.set_trace()
        losses.update(loss_triplet.item())
        loss.backward()
        optimizer.step()
        # lr.update(optimizer.param_groups[0]['lr'])
        lr.update(scheduler.get_lr()[0])
        bar.set_description(f'[Epoch {epoch}] '
                            f'lr: {lr.val:4f}, '
                            f'loss: {losses.val:.4f}({losses.avg:.4f}), ')
        bar.update()

    bar.close()

    logger.info(f'[EPOCH {epoch}] '
                f'lr: {lr.val:4f}, '
                f'loss: {losses.avg:4f}, ')
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/learning_rate', lr.val, epoch)


def test(loader, net, criterion, epoch):
    losses = AverageMeter()
    bar = tqdm(loader, ncols=150)
    # switch to evaluation mode
    net.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        _output1, _output2, _output3 = net(data1.float().cuda(), data2.float().cuda(), data3.float().cuda())
        dist_a = F.pairwise_distance(_output1, _output2, 2)
        dist_b = F.pairwise_distance(_output1, _output3, 2)
        target = torch.FloatTensor(dist_a.size()).fill_(1).cuda()
        target = Variable(target)
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = _output1.norm(2) + _output2.norm(2) + _output3.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        # loss = criterion(_output1, _output2, _output3)
        losses.update(loss_triplet.item())
        bar.set_description(f'[Epoch {epoch}] '
                            f'test loss: {losses.val:.4f}({losses.avg:.4f}), ')
        bar.update()

    bar.close()
    writer.add_scalar('test/loss', losses.avg, epoch)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN model.")
    parser.add_argument('--model', type=str, choices=models.FRAME_MODELS, default='Resnet50_RMAC')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--triplet_csv', type=str, required=False)
    parser.add_argument('--fivr_root', type=str, required=False)
    parser.add_argument('--vcdb_root', type=str, required=False)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-m', '--margin', type=float, default=0.3)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-tb', '--test_batch', type=int, default=256)
    parser.add_argument('-w', '--worker', type=int, default=4)
    parser.add_argument('-o', '--optim', type=str, default='adam')
    parser.add_argument('-s', '--scheduler', type=str, default='cyclic')

    parser.add_argument('-l', '--log_dir', type=str, default='./log')
    parser.add_argument('-c', '--comment', type=str, default=None)
    args = parser.parse_args()

    args.model = 'Resnet50_avgpool_FC3'
    args.triplet_csv = '/mldisk/nfs_shared_/dh/VCDB/vcdb_triplets_1fps.csv'
    args.fivr_root = '/mldisk/nfs_shared_/MLVD/FIVR'
    args.vcdb_root = '/mldisk/nfs_shared_/MLVD/VCDB'
    image_size = 28
    # args.scheduler = None
    writer, logger, log_dir = initialize_writer_and_log(args.log_dir, args.comment)
    model_save_dir = os.path.join(log_dir, 'saved_model')
    os.makedirs(model_save_dir)
    logger.info(args)

    # Model

    embed_net = models.get_frame_model(args.model).cuda()
    # embed_net = Net()
    # Load checkpoints
    if args.model == 'Resnet50_avgpool_FC3':
        embed_net.Resnet50_avgpool.requires_grad = False
    if args.ckpt is not None:
        embed_net.load_state_dict(torch.load(args.ckpt))

    model = models.TripletNet(embed_net).cuda()


    writer.add_graph(model, [torch.rand((2, 3, image_size, image_size)).cuda(),
                             torch.rand((2, 3, image_size, image_size)).cuda(),
                             torch.rand((2, 3, image_size, image_size)).cuda()])
    logger.info(model)

    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    # Data
    transform = {
        'train': trn.Compose([
            trn.Resize((image_size, image_size)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': trn.Compose([
            trn.Resize((image_size, image_size)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    vcdb = datasets.get_dataset('VCDB', args.vcdb_root)
    # fivr = datasets.get_dataset('FIVR', args.fivr_root)

    # train_loader = DataLoader(datasets.TripletDataset(args.triplet_csv, vcdb.frame_root, transform=transform['train']),
    #                           shuffle=True, num_workers=args.worker, batch_size=args.batch)
    # test_loader = DataLoader(datasets.ListDataset([], transform=transform['test']),
    #                          shuffle=False, num_workers=args.worker, batch_size=args.test_batch)
    train_loader = torch.utils.data.DataLoader(
        MNIST_t('/workspace/data/mnist', train=True, download=True,transform=transform['train']),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        MNIST_t('/workspace/data/mnist', train=False, transform= transform['test']),
        batch_size=512, shuffle=True)

    # Optimizer
    # criterion = nn.TripletMarginLoss(args.margin)
    criterion = nn.MarginRankingLoss(margin=args.margin)
    l2_dist = nn.PairwiseDistance()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'radam':
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35,40,45], gamma=0.8)

    # eval_vcdb_frame_rank(model, test_loader, vcdb, 0, 1000, 1)
    for e in range(1, args.epoch, 1):
        train(model, train_loader, optimizer, criterion,scheduler, e)

        test(test_loader, model, criterion, e)
        scheduler.step()
        torch.save({'epoch': e,
                    'state_dict': model.module.embedding_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(model_save_dir, f'epoch_{e}_ckpt.pth'))