import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datas.texture import DBreader_DynTex, DBreader_SynTex, DBreader_BVItexture, Sampler, HomTex
from configs.config import learning_rate, num_epochs, model_save_path, device_id, mode, delta
from util.utils import adjust_learning_rate
from model.GDConvNet import L1_Charbonnier_loss, Net
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='fine tune GDConvNet on texture databases')
    parser.add_argument('--texture', type=str, help='Path of the dataset.')
    parser.add_argument('--out_dir', type=str, help='Name of sequence.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)

    args = parser.parse_args()

    learning_rate = args.lr
    num_epochs = args.epochs

    args.out_dir = args.out_dir + '/finetune_{}'.format(args.texture)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Choose Gpu device
    device_ids = device_id
    device = torch.device("cuda:{}".format(device_id[0]) if torch.cuda.is_available() else "cpu")

    # Build model
    net = Net(nf=144, growth_rate=2, mode=mode)


    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # multi-GPU
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids[:1])

    # calculate all trainable parameters in network
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    dyntex_dir = '/mnt/storage/home/mt20523/scratch/DynTex'
    syntex_dir = '/mnt/storage/home/mt20523/scratch/SynTex'
    bvitexture_dir = '/mnt/storage/home/mt20523/scratch/BVI-Texture'
    homtex_dir = '/mnt/storage/home/mt20523/scratch/HomTex'
    dataset_dyntex = DBreader_DynTex(dyntex_dir, args.texture, random_crop=(256, 256))
    dataset_syntex = DBreader_SynTex(syntex_dir, args.texture, random_crop=(256, 256))
    dataset_bvitexture = DBreader_BVItexture(bvitexture_dir, args.texture, random_crop=(256, 256))
    sampler = Sampler([dataset_dyntex, dataset_syntex, dataset_bvitexture])

    train_loader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=True)
    test_loader = HomTex(homtex_dir, texture='mixed')


    print(len(train_loader))

    # Load Network weight
    net.load_state_dict(torch.load(model_save_path + 'net_best_weight'), strict=True)
    print("Best weight has been loaded")

    cb_loss = L1_Charbonnier_loss()
    cb_loss = cb_loss.to(device)

    # start training
    for epoch in range(0, num_epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        # epoch training start
        for batch_id, train_data in enumerate(train_loader):
            # initialize network and optimizer parameter gradients
            net.train()
            net.zero_grad()
            optimizer.zero_grad()

            img1, img2, img4, img5, gt = train_data
            img1 = img1.to(device)
            img2 = img2.to(device)
            img4 = img4.to(device)
            img5 = img5.to(device)
            gt = gt.to(device)


            # forward + backward + optimize
            oup, mid_oup = net(img1, img2, img4, img5)

            oup_loss = cb_loss(oup, gt)
            mid_oup_loss = cb_loss(mid_oup, gt)

            # perceptual_loss = loss_network(oup, gt)
            loss = oup_loss + delta * mid_oup_loss
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            # print out
            if not (batch_id % 10):
                if batch_id == 0 and epoch !=0:
                    continue
                print('Epoch:{0}, Iteration:{1}, central_psnr:{2:.2f}'.format(epoch, batch_id))

        # Average PSNR on one epoch train_data
        train_one_epoch_time = time.time() - start_time
        print("Training one epoch costs {}s".format(train_one_epoch_time))

        # use evaluation model during the net evaluating
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch+1, 'state_dict': net.state_dict()}, ckpt_dir+'/model_epoch'+str(epoch+1).zfill(3)+'.pth')
            net.eval()
            test_loader.Test(net, epoch+1, result_dir)




if __name__ == "__main__":
    main()
