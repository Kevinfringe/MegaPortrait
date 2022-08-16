import argparse
import torch
import dataset
import model
import cv2 as cv
import HeadPoseEstimation
import vgg_face
import numpy as np
import torch.nn as nn
import patchGAN
import random
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR


source_img_path = './source_img'
driver_img_path = './driver_img'
data_set_length = 42
img_size = 512

hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
L1_loss = nn.L1Loss(reduction='mean')
feature_matching_loss = nn.MSELoss()
cosine_dist = nn.CosineSimilarity()

patch = (1, img_size // 2 ** 4, img_size // 2 ** 4)

def cosine_distance(args, z1, z2):

    res = args.s_cos * (torch.sum(cosine_dist(z1[0], z2[0])) - args.m_cos)
    res += args.s_cos * (torch.sum(cosine_dist(z1[1], z2[1])) - args.m_cos)
    res += args.s_cos * (torch.sum(cosine_dist(z1[2], z2[2])) - args.m_cos)

    return res


def cosine_loss( args, descriptor_driver,
                descriptor_source_rand, descriptor_driver_rand):

    z_dri = descriptor_driver
    z_dri_rand = descriptor_driver_rand

    # Create descriptors to form the pairs
    # z_s*->d
    z_src_rand_dri = [descriptor_source_rand[0],
                      z_dri[1],
                      z_dri[2]]

    # Form the pairs
    pos_pairs = [(z_dri, z_dri), (z_src_rand_dri, z_dri)]
    neg_pairs = [(z_dri, z_dri_rand), (z_src_rand_dri, z_dri_rand)]

    # Calculate cos loss
    sum_neg_paris = torch.exp(cosine_distance(args, neg_pairs[0][0], neg_pairs[0][1])) + torch.exp(cosine_distance(args, neg_pairs[1][0], neg_pairs[1][1]))

    L_cos = torch.zeros(dtype=torch.float)
    for i in range(len(pos_pairs)):
        L_cos += torch.log(torch.exp(cosine_distance(args, pos_pairs[0][0], pos_pairs[0][1])) / (torch.exp(cosine_distance(args, pos_pairs[0][0], pos_pairs[0][1])) + sum_neg_paris))

    return L_cos



def train(args, models, device, driver_loader, source_loader, optimizers, schedulers, source_img_random, driver_img_random):
    Eapp1 = models['Eapp1']
    Eapp2 = models['Eapp2']
    Emtn_facial = models['Emtn_facial']
    Warp_G = models['Warp_G']
    G3d = models['G3d']
    G2d = models['G2d']
    vgg_IN = models['Vgg_IN']
    vgg_face = models['Vgg_face']
    discriminator = models['patchGAN']

    train_loss = 0.0


    for idx in range(args.iteration):
        # Ending condition for training.
        if idx > args.iteration:
            print(" Training complete!")

            break
        else:
            idx += 1

        # loading a single data
        source_img = next(source_loader).to(device)
        driver_img = next(driver_loader).to(device)

        # pass the data through Eapp1 & 2.
        v_s = Eapp1(source_img)
        e_s = Eapp2(source_img)

        # Emtn.

        # First part of Emtn : Generate the transformation matrix
        # based on the head pose estimation.
        trans_mat_source = HeadPoseEstimation.head_pose_estimation(source_img)
        trans_mat_driver = HeadPoseEstimation.head_pose_estimation(driver_img)

        # Second part of Emtn : Generate facial expression letent vector z
        # based on a ResNet-18 network
        z_s = Emtn_facial(source_img)
        z_d = Emtn_facial(driver_img)

        # Warp_Generator

        # First part of Warp Generator: Generate warping matrix
        # based on its transformation matrix.
        W_rt_s = np.linalg.inv(trans_mat_source)
        W_rt_d = trans_mat_driver

        # Second part of Warp Generator: Generate emotion warper.
        W_em_s = Warp_G(z_s + e_s)
        W_em_d = Warp_G(z_d + e_s)

        # 3D warping of w_s and v_s
        # First, 3D warping using w_rt_s
        warp_3d_vs = cv.warpPerspective(v_s, W_rt_s, (v_s.shape[1], v_s.shape[0]))
        # Next, 3D warping using w_em_s
        warp_3d_vs = cv.warpPerspective(warp_3d_vs, W_em_s, (warp_3d_vs.shape[1], warp_3d_vs.shape[0]))

        # Pass data into G3d
        output = G3d(warp_3d_vs)

        # 3D warping with w_d
        vs_d = cv.warpPerspective(warp_3d_vs, W_rt_d, (warp_3d_vs.shape[1], warp_3d_vs.shape[0]))
        vs_d = cv.warpPerspective(vs_d, W_em_d, (vs_d.shape[1], vs_d.shape[0]))

        # Pass into G2d.
        output = G2d(vs_d)

        # IN loss
        L_IN = L1_loss(vgg_IN(output), vgg_IN(driver_img))

        # face loss
        L_face = L1_loss(vgg_face(output), vgg_face(driver_img))

        # adv loss
        # Adversarial ground truths
        valid = Variable(torch.Tensor(np.ones((driver_img.size(0), *patch))), requires_grad=False)
        fake = Variable(torch.Tensor(-1 * np.ones((driver_img.size(0), *patch))), requires_grad=False)

        # real loss
        pred_real = discriminator(driver_img, source_img)
        loss_real = hinge_loss(pred_real, valid)

        # fake loss
        pred_fake = discriminator(output.detach(), source_img)
        loss_fake = hinge_loss(pred_fake, fake)

        L_adv = 0.5 * (loss_real + loss_fake)

        # feature mapping loss
        L_feature_matching = feature_matching_loss(output, driver_img)

        # Cycle consistency loss
        # Feed base model with randomly sampled image.
        e_s_rand = Eapp2(source_img_random)
        trans_mat_source_rand = HeadPoseEstimation.head_pose_estimation(source_img_random)
        z_s_rand = Emtn_facial(source_img_random)

        trans_mat_driver_rand = HeadPoseEstimation.head_pose_estimation(driver_img_random)
        z_d_rand = Emtn_facial(driver_img_random)

        descriptor_driver = [e_s, trans_mat_driver, z_d]
        descriptor_source_rand = [e_s_rand, trans_mat_source_rand, z_s_rand]
        descriptor_driver_rand = [e_s_rand, trans_mat_driver_rand, z_d_rand]

        L_cos = cosine_loss(args, descriptor_driver,
                           descriptor_source_rand, descriptor_driver_rand)

        L_per = args.weight_IN * L_IN + args.weight_face * L_face

        L_gan = args.weight_adv * L_adv + args.weight_FM * L_feature_matching

        L_final = L_per + L_gan + args.weight_cos * L_cos

        # Optimizer and Learning rate scheduler.
        # optimizer
        for i in range(len(optimizers)):
            optimizers[i].zero_grad()

        L_final.backward()

        for i in range(len(optimizers)):
            optimizers[i].step()
            schedulers[i].step()

        train_loss += L_final
        train_loss /= idx

        # Print log
        print('Iteration: {} / {} : train loss is: {}'.format(idx, args.iteration, train_loss))




def main():
    parser = argparse.ArgumentParser(description="Megaportrait Pytorch implementation.")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training. default=16')
    parser.add_argument('--iteration', type=int, default=20000, metavar='N',
                        help='input batch size for training. default=20000')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train. default = 50')
    parser.add_argument('--weight-IN', type=int, default=20, metavar='N',
                        help='weight parameter for IN loss. default = 20')
    parser.add_argument('--weight-face', type=int, default=5, metavar='N',
                        help='weight parameter for face loss. default = 5')
    parser.add_argument('--weight-adv', type=int, default=1, metavar='N',
                        help='weight parameter for adv loss. default = 1')
    parser.add_argument('--weight-FM', type=int, default=40, metavar='N',
                        help='weight parameter for feature matching loss. default = 40')
    parser.add_argument('--weight-cos', type=int, default=2, metavar='N',
                        help='weight parameter for cos loss. default = 2')
    parser.add_argument('--s-cos', type=int, default=5, metavar='N',
                        help='s parameter in cos loss. default = 5')
    parser.add_argument('--m-cos', type=float, default=0.2, metavar='N',
                        help='m parameter in cos loss. default = 0.2')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Transformation on source images.
    transform_source = transforms.Compose([
        transforms.ToTensor(),
        # Normalize data into range(-1, 1)
        transforms.Normalize([0.5], [0.5]),
        # Randomly flip train data(left and right).
        transforms.RandomHorizontalFlip(),
        # Color jitter on data.
        transforms.ColorJitter()
    ])

    # Transformation on driver images.
    transform_driver = transforms.Compose([
        transforms.ToTensor(),
        # Normalize data into range(-1, 1)
        transforms.Normalize([0.5], [0.5]),
        # Randomly flip train data(left and right).
        transforms.RandomHorizontalFlip(),
        # Color jitter on data.
        transforms.ColorJitter()
    ])

    # Define dataset loaders
    source_set = dataset.SourceDataset(source_img_path, data_set_length, transform=transform_source)
    driver_set = dataset.DriverDataset(driver_img_path, data_set_length, transform=transform_driver)

    source_loader = DataLoader(source_set, batch_size=args.batch_size, shuffle=False)
    driver_loader = DataLoader(driver_set, batch_size=args.batch_size, shuffle=False)

    # Generate random pairs for calculating cos loss.
    random.seed(0)
    [index_source, index_driver] = random.sample(range(0, 29999), 2)
    source_img_random = Image.open("./CelebA-HQ-img/" + str(index_source) + ".jpg")
    driver_img_random = Image.open("./CelebA-HQ-img/" + str(index_driver) + ".jpg")
    # Apply the same transformation on these two images.
    source_img_random = transform_source(source_img_random)
    driver_img_random = transform_driver(driver_img_random)

    Eapp1 = model.Eapp1().to(device)
    Eapp2 = model.Eapp2().to(device)
    Emtn_facial = model.Emtn_facial().to(device)
    Emtn_head = model.Emtn_head().to(device)
    Warp_G = model.WarpGenerator().to(device)
    G3d = model.G3d().to(device)
    G2d = model.G2d().to(device)
    Vgg_IN = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    Vgg_face = vgg_face.VggFace().to(device)
    discriminator = patchGAN.Discriminator().to(device)

    models = {
        'Eapp1': Eapp1,
        'Eapp2': Eapp2,
        'Emtn_facial': Emtn_facial,
        'Emtn_head': Emtn_head,
        'Warp_G': Warp_G,
        'G3d': G3d,
        'G2d': G2d,
        'Vgg_IN': Vgg_IN,
        'Vgg_face': Vgg_face,
        'patchGAN': discriminator
    }

    optimizers = [
        optim.Adam(models['Eapp1'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Eapp2'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Emtn_facial'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Emtn_head'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Warp_G'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['G3d'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['G2d'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['patchGAN'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2)
    ]

    schedulers = []

    for i in range(len(optimizers)):
        scheduler = CosineAnnealingLR(optimizers[0], T_max=args.iteration, eta_min=1e-6)
        schedulers.append(scheduler)

    train(args, args, models, device, driver_loader, source_loader, optimizers, schedulers, source_img_random, driver_img_random)


# Start Training.
main()