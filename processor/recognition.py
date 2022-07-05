#!/usr/bin/env python
# pylint: disable=W0201
from pyexpat import features
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss_1 = nn.CrossEntropyLoss()
        self.loss_2 = nn.MSELoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            self.optimizer_2 = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)    
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss_1(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    # MTGEA 2 stream train 
    def doubletrain(self):
        self.model.train()    
        self.adjust_lr()    
        loader = self.data_loader['first train']
        loader_2 = self.data_loader['second train']    
        loss_value = []
        loss_2_value = []
        loss_3_value = []

        for first, second in zip(loader, loader_2):

            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
            second[1] = second[1].long().to(self.dev)    


            # forward
            output  = self.model(first[0], second[0])
            loss_1 = self.loss_1(output, first[1])


            # backward
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss_1.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
 

    def test(self, evaluation=True):
        correct=0
        total_data = 0
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        # result_frag = []
        label_frag = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # print("test label: ", label)

            # inference
            with torch.no_grad():
                output = self.model(data)
                _, output_index = torch.max(output, 1)  
                # print("prediction: ", output_index)
               
                if evaluation:
                    for i in range(len(output_index)):
                        if (output_index[i] == label[i]):
                            correct +=1
                    total_data += len(output_index)
                
                    
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        return correct, (100*(correct/total_data))

    # MTGEA 2 stream train 
    def doubletest(self, evaluation=True):
        self.relu = torch.nn.ReLU()
        correct = 0
        total_data = 0
        self.model.eval()  
        loader = self.data_loader['first test']
        loader_2 = self.data_loader['second test']
        loss_value = []
        label_frag = []
        result_frag = []

        for first, second in zip(loader, loader_2):
      
            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
    
            
            # print("test label: ", first[1])

            # inference
            with torch.no_grad():
                output = self.model(first[0], second[0])
                _, final_output_index = torch.max(output, 1)
                # print("prediction: ", final_output_index)


            if evaluation:
                for i in range(len(final_output_index)):
                    if (final_output_index[i] == first[1][i]):
                        correct +=1
                total_data += len(final_output_index)       
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        return correct, (100*(correct/total_data))   


    # MTGEA freezing train 
    def freezingtrain(self):
        self.model.train()    
        self.adjust_lr()    
        loader = self.data_loader['first train']
        loader_2 = self.data_loader['second train']    
        loss_value = []
        for name, child in self.model.named_children():

            for param in child.parameters():
                if name == 'kin_stgcn':
                    param.requires_grad = False
 

        for first, second in zip(loader, loader_2):

            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
            second[1] = second[1].long().to(self.dev)    

            # forward
            output  = self.model(first[0], second[0])
            loss_1 = self.loss_1(output, first[1])

            # backward
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss_1.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    # MTGEA freezing test
    def freezingtest(self, evaluation=True):
        self.relu = torch.nn.ReLU()
        correct = 0
        total_data = 0
        self.model.eval()  
        loader = self.data_loader['first test']
        loader_2 = self.data_loader['second test']
        loss_value = []
        label_frag = []
        result_frag = []

        # check for parameter
        """
        for name, child in self.model.named_children():
            for param in child.parameters():
                if name == 'kin_stgcn':
                    print("kin freezing prarm", param)
        """            

        for first, second in zip(loader, loader_2):
    
            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
    
            
            # print("test label: ", first[1])

            # inference
            with torch.no_grad():
                output = self.model(first[0], second[0])
                _, final_output_index = torch.max(output, 1)
                # print("prediction: ", final_output_index)


            if evaluation:
                for i in range(len(final_output_index)):
                    if (final_output_index[i] == first[1][i]):
                        correct +=1
                total_data += len(final_output_index)       
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        
        return correct, (100*(correct/total_data))   


    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser