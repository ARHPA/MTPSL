import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import model.config_task as config_task
import pdb

from transformers import SegformerForSemanticSegmentation, SegformerConfig

def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    return loss

class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1, num_tasks=2):
        super(conv_task, self).__init__()
        self.num_tasks = num_tasks
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.ones(planes, num_tasks*(num_tasks-1)))
        self.beta = nn.Parameter(torch.zeros(planes, num_tasks*(num_tasks-1)))
        self.bn = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):

        # first, get the taskpair information: compute A
        A_taskpair = config_task.A_taskpair

        x = self.conv(x)

        # generate taskpair-specific FiLM parameters
        gamma = torch.mm(A_taskpair, self.gamma.t())
        beta = torch.mm(A_taskpair, self.beta.t())
        gamma = gamma.view(1, x.size(1), 1, 1)
        beta = beta.view(1, x.size(1), 1, 1)

        x = self.bn(x)

        # taskpair-specific transformation
        x = x * gamma + beta
        x = self.relu(x)


        return x


class SegFormerMTL_enc(nn.Module):
    def __init__(self, input_channels):
        super(SegFormerMTL_enc, self).__init__()
        
        self.num_tasks = len(input_channels)
        self.input_channels = input_channels
        
        # Initialize SegFormer backbone with lightweight configuration
        config = SegformerConfig(
            num_channels=3,
            num_labels=150,  # Use standard ADE20K labels for pretrained model
            image_size=512,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],  # Lightweight configuration
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],  # Smaller hidden sizes for lightweight variant
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=0.1,
            layer_norm_eps=1e-6,
            decoder_hidden_size=256,
            semantic_loss_ignore_index=-1,
        )
        
        # Load pretrained SegFormer backbone
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Get the encoder for feature extraction
        self.encoder = self.segformer.segformer.encoder
        
        # Task-specific input preprocessing layers
        self.pred_encoder_source = nn.ModuleList()
        for i in range(len(input_channels)):
            if input_channels[i] == 3:
                # For RGB input, use identity mapping
                self.pred_encoder_source.append(nn.Identity())
            else:
                # For other inputs, project to 3 channels to match SegFormer input
                self.pred_encoder_source.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels[i], 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 3, kernel_size=3, padding=1),
                        nn.BatchNorm2d(3),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # Feature dimension from SegFormer encoder
        self.feature_dim = 256  # Last hidden size from config
        
        # Task-conditioned feature projection
        self.task_projections = nn.ModuleList()
        for i in range(self.num_tasks):
            self.task_projections.append(
                nn.Sequential(
                    nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1),
                    nn.BatchNorm2d(self.feature_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Initialize custom layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.pred_encoder_source:
            if isinstance(m, nn.Sequential):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.xavier_normal_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
        
        for m in self.task_projections:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, input_task):
        batch_size, _, input_h, input_w = x.shape
        
        # Task-specific input preprocessing
        x_processed = self.pred_encoder_source[input_task](x)
        
        # Extract features using SegFormer encoder
        encoder_outputs = self.encoder(
            x_processed,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the last hidden state (highest level features)
        # encoder_outputs.hidden_states contains features from all stages
        # Use the last stage features for the deepest representation
        last_hidden_state = encoder_outputs.hidden_states[-1]
        
        # Apply task-specific projection
        task_features = self.task_projections[input_task](last_hidden_state)
        
        return task_features
    
    def get_multi_scale_features(self, x, input_task):
        """
        Alternative method to get multi-scale features from all encoder stages
        Useful for feature pyramid or multi-scale processing
        """
        batch_size, _, input_h, input_w = x.shape
        
        # Task-specific input preprocessing
        x_processed = self.pred_encoder_source[input_task](x)
        
        # Extract features using SegFormer encoder
        encoder_outputs = self.encoder(
            x_processed,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get features from all stages
        multi_scale_features = []
        for i, hidden_state in enumerate(encoder_outputs.hidden_states):
            # Apply task-specific projection to each scale
            if i < len(self.task_projections):
                projected_features = self.task_projections[input_task](hidden_state)
            else:
                # For stages beyond our projections, use the last projection
                projected_features = self.task_projections[input_task](hidden_state)
            multi_scale_features.append(projected_features)
        
        return multi_scale_features

class SegNet_enc(nn.Module):
    def __init__(self, input_channels):
        super(SegNet_enc, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.filter = filter
        self.num_tasks = len(input_channels)
        # Task-specific input layer
        self.pred_encoder_source = nn.ModuleList([self.pre_conv_layer([input_channels[0], filter[0]])])
        for i in range(1, len(input_channels)):
            self.pred_encoder_source.append(self.pre_conv_layer([input_channels[i], filter[0]]))

        # define shared mapping function, which is conditioned on the taskpair
        self.encoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(len(filter)-1):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(len(filter)-1):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x, input_task):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * len(self.filter) for _ in range(5))
        for i in range(len(self.filter)):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        
        # task-specific input layer
        # if input_task is not None:
        x = self.pred_encoder_source[input_task](x)

        # shared mapping function
        for i in range(len(self.filter)):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        return g_maxpool[-1]


    def conv_layer(self, channel):
        return conv_task(in_planes=channel[0], planes=channel[1], num_tasks=self.num_tasks)

    def pre_conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True)
        )
        return conv_block


class Mapfns(nn.Module):
    def __init__(self, tasks=['semantic', 'depth', 'normal'], input_channels=[13, 1, 3]):
        super(Mapfns, self).__init__()
        # initialise network parameters
        assert len(tasks) == len(input_channels) 
        self.tasks = tasks 
        self.input_channels = {}
        for t, task in enumerate(tasks):
            self.input_channels[task] = input_channels[t]

        self.mapfns = SegFormerMTL_enc(input_channels=input_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_pred, gt, feat, w, ssl_type, reg_weight=0.5):
        if ssl_type == 'full':
            target_task_index = torch.arange(len(self.tasks)) 
            source_task_index = torch.arange(len(self.tasks))
        else:
            target_task_index = (w.data == 1).nonzero(as_tuple=False).view(-1)
            source_task_index = (w.data == 0).nonzero(as_tuple=False).view(-1)

        # loss = torch.tensor(0).to(feat.device)
        loss = 0
        if len(source_task_index) > 0:
            for source_task in source_task_index:
                for target_task in target_task_index:
                    if source_task != target_task:

                        source_pred = x_pred[source_task]
                        target_gt = gt[target_task]

                        source_pred = self.pre_process_pred(source_pred, task=self.tasks[source_task])
                        target_gt = self.pre_process_gt(target_gt, task=self.tasks[target_task])

                        # config_task.source_task = [source_task]
                        # config_task.target_task = [target_task]
                        # source_task, target_task = config_task.source_task[0], config_task.target_task[0]
                        A_taskpair = torch.zeros(len(self.tasks), len(self.tasks)).to(source_pred.device)
                        A_taskpair[source_task, target_task] = 1.0
                        n, m = A_taskpair.size()
                        A_taskpair = A_taskpair.flatten()[:-1].view(n-1,n+1)[:,1:].flatten().view(1,-1)
                        config_task.A_taskpair = A_taskpair
                        
                        print(f"source_pred: {source_pred.shape}")
                        print(f"target_gt: {target_gt.shape}")

                        mapout_source = self.mapfns(source_pred, input_task=source_task)
                        mapout_target = self.mapfns(target_gt, input_task=target_task)

                        print(f"mapout_source: {mapout_source.shape}")
                        print(f"mapout_target: {mapout_target.shape}")
                        print(f"feat: {feat.detach().shape}")

                        loss = loss + self.compute_loss(mapout_source, mapout_target, feat, reg_weight=reg_weight)
                 

        return loss

    def pre_process_pred(self, pred, task):
        if task == 'semantic':
            x_pred = F.gumbel_softmax(pred, dim=1, tau=1, hard=True)
            while torch.isnan(x_pred.sum()):
                x_pred = F.gumbel_softmax(pred, dim=1, tau=1, hard=True)
            pred = x_pred
        elif task == 'depth':
            x_pred = pred / (pred.max() + 1e-12) 
            pred = x_pred
        elif task == 'normal':
            pred = (pred + 1.0) / 2.0
        return pred

    def pre_process_gt(self, gt, task):
        if task == 'semantic':
            gt = gt.unsqueeze(0)
            binary_mask = (gt == -1).type(torch.FloatTensor).cuda()
            num_classes = self.input_channels[task]
            gt_ = gt.float() * (1 - binary_mask)
            gt__ = torch.zeros(gt.size(0), num_classes, gt.size(2), gt.size(3)).scatter_(1, gt_.type(torch.LongTensor), 1).cuda().detach() * (1 - binary_mask)
        elif task == 'depth':
            gt__ = gt / (gt.max() + 1e-12)
        else:
            gt__ = (gt + 1.0) / 2.0
            # gt__ = gt
        return gt__

    def compute_loss(self, mapout_source, mapout_target, feat, reg_weight=0.5):
        # cross-task consistency
        l_s_t = 1 - F.cosine_similarity(mapout_source, mapout_target, dim=1, eps=1e-12).mean()
        # regularization
        l_s_f = 1 - F.cosine_similarity(mapout_source, feat.detach(), dim=1, eps=1e-12).mean()
        l_t_f = 1 - F.cosine_similarity(mapout_target, feat.detach(), dim=1, eps=1e-12).mean()

        if reg_weight > 0:
            loss = l_s_t + reg_weight * (l_s_f + l_t_f)
        else:
            loss = l_s_t
        return loss

 