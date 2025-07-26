#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from model.N2NUnet import N2NUnet
from model.utils import *

import os
import json
from datetime import datetime



class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model selection
        if self.p["noise_type"] == 'mc':
            self.is_mc = True
            self.model = N2NUnet(in_channels=3)
        else:
            self.is_mc = False
            self.model = N2NUnet(in_channels=3)

        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=float(self.p["learning_rate_cli"]))

            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optim,
                patience=self.p["nb_epochs"] // 4,
                factor=0.5,
                verbose=True)

            if self.p["loss"] == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif self.p["loss"] == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        self.use_cuda = torch.cuda.is_available() and self.p["cuda"]
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()

    def _print_params(self):
        print('Training parameters: ')
        self.p["cuda"] = self.use_cuda
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in self.p.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        from datetime import datetime

        if first:
            if self.p["clean_targets"]:
                ckpt_dir_name = f'{datetime.now():{self.p["noise_type"]}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p["noise_type"]}-%H%M}'
            if self.p["ckpt_overwrite"]:
                ckpt_dir_name = f'{self.p["noise_type"]}-clean' if self.p["clean_targets"] else self.p["noise_type"]

            self.ckpt_dir = os.path.join(self.p["ckpt_save_path"], ckpt_dir_name)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.p["ckpt_overwrite"]:
            fname_unet = f'{self.ckpt_dir}/n2n-{self.p["noise_type"]}.pt'
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = f'{self.ckpt_dir}/n2n-epoch{epoch + 1}-{valid_loss:.5f}.pt'

        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        with open(f'{self.ckpt_dir}/n2n-stats.json', 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        print('Loading checkpoint from: {}'.format(ckpt_fname))
        state_dict = torch.load(ckpt_fname, map_location='cuda' if self.use_cuda else 'cpu')
        self.model.load_state_dict(state_dict)

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        self.scheduler.step(valid_loss)

        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        if self.p["plot_stats"]:
            loss_str = f'{self.p["loss"].upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    def test(self, test_loader, show):
        self.model.train(False)
        source_imgs, denoised_imgs, clean_imgs = [], [], []

        denoised_dir = os.path.dirname(self.p["data"])
        save_path = os.path.join(denoised_dir, 'denoised')
        os.makedirs(save_path, exist_ok=True)

        batch_idx = 0
        for batch in test_loader:
            source = batch['input']
            target = batch['noisy']
            if show == 0 or batch_idx >= show:
                break
            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            denoised_img = self.model(source).detach()
            denoised_imgs.append(denoised_img)

            batch_idx+=1

        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        print('Saving images and montages to: {}'.format(save_path))
        for i in range(len(source_imgs)):
            img_name = "name"
            create_montage(img_name, self.p["noise_type"], save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show)

    def eval(self, valid_loader):
        self.model.train(False)
        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch in valid_loader:
            source = batch['input']
            target = batch['noisy']
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            source_denoised = self.model(source)
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            if self.is_mc:
                source_denoised = reinhard_tonemap(source_denoised)

            for i in range(self.p["batch_size_cli"]):
                source_denoised = source_denoised.cpu()
                target = target.cpu()
                psnr_meter.update(psnr(source_denoised[i], target[i]).item())

        return loss_meter.avg, time_elapsed_since(valid_start)[0], psnr_meter.avg

    def train(self, train_loader, valid_loader):
        self.model.train(True)
        self._print_params()
        num_batches = len(train_loader)

        assert num_batches % self.p["report_interval"] == 0, 'Report interval must divide total number of batches'

        stats = {
            'noise_type': self.p["noise_type"],
            'noise_param': self.p["noise_param"],
            'train_loss': [],
            'valid_loss': [],
            'valid_psnr': []
        }

        train_start = datetime.now()
        for epoch in range(self.p["nb_epochs"]):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p["nb_epochs"]))

            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            batch_idx = 0
            for batch in valid_loader:
                source = batch['input']
                target = batch['noisy']
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p["report_interval"], loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                source_denoised = self.model(source)
                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p["report_interval"] == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
                batch_idx += 1

            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        print('Training done! Total elapsed time: {}\n'.format(time_elapsed_since(train_start)[0]))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
