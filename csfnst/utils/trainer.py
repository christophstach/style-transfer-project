import os.path
import warnings
from time import time

import math
import torch
import torch.autograd
import torch.cuda
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, device, config, model, criterion, dataloader, data_transformer=None, tensorboard=True):
        super().__init__()

        config['running_loss_range'] = config['running_loss_range'] if 'running_loss_range' in config else 50

        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.config = config

        self.model = model.to(device)
        self.criterion = criterion

        self.ds_length = len(dataloader.dataset)
        self.batch_size = dataloader.batch_size
        self.dataloader = dataloader
        self.data_transformer = data_transformer

        self.epoch_loss_history = []
        self.content_loss_history = []
        self.style_loss_history = []
        self.total_variation_loss_history = []
        self.loss_history = []
        self.lr_history = []

        self.progress_bar = trange(
            math.ceil(self.ds_length / self.batch_size) * self.config['epochs'],
            leave=True
        )

        if self.config['lr_scheduler'] == 'CyclicLR':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config['max_lr'], nesterov=True, momentum=0.9)
            self.scheduler = CyclicLR(
                optimizer=self.optimizer,
                base_lr=config['min_lr'],
                max_lr=config['max_lr'],
                step_size_up=self.config['lr_step_size'],
                mode='triangular2'
            )
        elif self.config['lr_scheduler'] == 'CosineAnnealingLR':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['max_lr'])
            self.scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.config['lr_step_size']
            )
        elif self.config['lr_scheduler'] == 'ReduceLROnPlateau':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['max_lr'])
            self.scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                patience=100
            )
        elif self.config['lr_scheduler'] == 'StepLR':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['max_lr'])
            self.scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=self.config['lr_step_size'],
                gamma=float(self.config['lr_multiplicator'])
            )
        elif self.config['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['max_lr'])
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=self.config['lr_step_size'],
                eta_min=config['min_lr'],
                T_mult=int(self.config['lr_multiplicator'])
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['max_lr'])
            self.scheduler = None

        if tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join('../runs', config['name']))
        else:
            self.tensorboard_writer = None

    def load_checkpoint(self):
        name = self.config['name']
        path = f'../checkpoints/{name}.pth'

        if os.path.exists(path):
            checkpoint = torch.load(path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.content_loss_history = checkpoint['content_loss_history']
            self.style_loss_history = checkpoint['style_loss_history']
            self.total_variation_loss_history = checkpoint['total_variation_loss_history']
            self.loss_history = checkpoint['loss_history']
            self.lr_history = checkpoint['lr_history']

            del checkpoint
            torch.cuda.empty_cache()

    def train(self):
        start = time()

        for epoch in range(self.config['epochs']):
            self.epoch_loss_history = []

            for i, batch in enumerate(self.dataloader):
                self.do_training_step(batch)
                self.do_progress_bar_step(epoch, self.config['epochs'], i)

                if self.config['lr_scheduler'] == 'ReduceLROnPlateau':
                    self.scheduler.step(self.loss_history[-1])
                elif self.scheduler:
                    self.scheduler.step()

                if i % self.config['save_checkpoint_interval'] == 0:
                    self.save_checkpoint(f'../checkpoints/{self.config["name"]}.pth')

                if time() - start >= self.config['max_runtime']:
                    break

        torch.cuda.empty_cache()

    def do_training_step(self, batch):
        self.model.train()

        with torch.autograd.detect_anomaly():
            try:

                if self.data_transformer:
                    x, y = self.data_transformer(batch)
                else:
                    x, y = batch

                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds = self.model(x)

                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                self.epoch_loss_history.append(self.criterion.loss_val)

                self.content_loss_history.append(self.criterion.content_loss_val)
                self.style_loss_history.append(self.criterion.style_loss_val)
                self.total_variation_loss_history.append(self.criterion.total_variation_loss_val)
                self.loss_history.append(self.criterion.loss_val)

                if self.tensorboard_writer:
                    grid_y = torchvision.utils.make_grid(y)
                    grid_preds = torchvision.utils.make_grid(preds)

                    self.tensorboard_writer.add_image('Inputs', grid_y, 0)
                    self.tensorboard_writer.add_image('Predictions', grid_preds, 0)

                    # writer.add_graph(network, images)
                    self.tensorboard_writer.add_scalar(
                        'Content Loss',
                        self.content_loss_history[-1],
                        len(self.content_loss_history) - 1
                    )

                    self.tensorboard_writer.add_scalar(
                        'Style Loss',
                        self.style_loss_history[-1],
                        len(self.style_loss_history) - 1
                    )

                    self.tensorboard_writer.add_scalar(
                        'TV Loss',
                        self.total_variation_loss_history[-1],
                        len(self.total_variation_loss_history) - 1
                    )

                    self.tensorboard_writer.add_scalar(
                        'Total Loss',
                        self.loss_history[-1],
                        len(self.loss_history) - 1
                    )

                    self.tensorboard_writer.add_scalar(
                        'Learning Rate',
                        self.lr_history[-1],
                        len(self.lr_history) - 1
                    )

                    self.tensorboard_writer.close()
            except:
                self.load_checkpoint()

    def do_validation_step(self):
        self.model.eval()

    def do_progress_bar_step(self, epoch, epochs, i):
        avg_epoch_loss = sum(self.epoch_loss_history) / (i + 1)

        if len(self.loss_history) >= self.config['running_loss_range']:
            running_loss = sum(
                self.loss_history[-self.config['running_loss_range']:]
            ) / self.config['running_loss_range']
        else:
            running_loss = 0

        if len(self.loss_history) > 0:
            self.progress_bar.set_description(
                f'Name: {self.config["name"]}, ' +
                f'Loss Network: {self.config["loss_network"]}, ' +
                f'Epoch: {epoch + 1}/{epochs}, ' +
                f'Average Epoch Loss: {avg_epoch_loss:,.2f}, ' +
                f'Running Loss: {running_loss:,.2f}, ' +
                f'Loss: {self.loss_history[-1]:,.2f}, ' +
                f'Learning Rate: {self.lr_history[-1]:,.6f}'
            )
        else:
            self.progress_bar.set_description(
                f'Name: {self.config["name"]}, ' +
                f'Loss Network: {self.config["loss_network"]}, ' +
                f'Epoch: {epoch + 1}/{epochs}, ' +
                f'Average Epoch Loss: {0:,.2f}, ' +
                f'Running Loss: {0:,.2f}, ' +
                f'Loss: {0:,.2f}, ' +
                f'Learning Rate: {0:,.6f}'
            )

        self.progress_bar.update(1)
        self.progress_bar.refresh()

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'content_loss_history': self.content_loss_history,
            'style_loss_history': self.style_loss_history,
            'total_variation_loss_history': self.total_variation_loss_history,
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
            'content_image_size': self.config['content_image_size'],
            'style_image_size': self.config['style_image_size'],
            'network': str(self.config['network']),
            'content_weight': self.config['content_weight'],
            'style_weight': self.config['style_weight'],
            'total_variation_weight': self.config['total_variation_weight'],
            'bottleneck_size': self.config['bottleneck_size'],
            'bottleneck_type': str(self.config['bottleneck_type']),
            'channel_multiplier': self.config['channel_multiplier'],
            'expansion_factor': self.config['expansion_factor'],
            'intermediate_activation_fn': self.config['intermediate_activation_fn'],
            'final_activation_fn': self.config['final_activation_fn']
        }, path)
