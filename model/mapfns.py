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
        # filter = [64, 128, 128, 256, 256]
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
    def __init__(self, tasks=['semantic', 'depth'], input_channels=[7, 1], 
                 backbone_feat_channels=512, embedding_dim=64, num_layers=3):
        super(Mapfns, self).__init__()
        self.tasks = tasks
        self.input_channels = {task: ch for task, ch in zip(tasks, input_channels)}
        
        # Shared lightweight encoder with FiLM conditioning
        self.encoder = LightweightMapEncoder(
            input_channels=input_channels,
            num_tasks=len(tasks),
            embedding_dim=embedding_dim,
            num_layers=num_layers
        )
        
        # Feature projection for backbone features - updated to 512 channels
        self.feat_proj = nn.Conv2d(backbone_feat_channels, 64, kernel_size=1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_pred, gt, feat, w, ssl_type='full', reg_weight=0.5):
        # Convert task indices to integers
        if ssl_type == 'full':
            source_indices = list(range(len(self.tasks)))
            target_indices = list(range(len(self.tasks)))
        else:
            # Convert to list of integers
            source_indices = (w == 0).nonzero(as_tuple=False).squeeze(1).tolist()
            target_indices = (w == 1).nonzero(as_tuple=False).squeeze(1).tolist()
        
        total_loss = 0
        valid_pairs = 0
        
        # Process all valid task pairs
        for s_idx in source_indices:
            for t_idx in target_indices:
                if s_idx == t_idx:
                    continue
                    
                # Preprocess predictions and ground truth
                source_pred = self.preprocess(x_pred[s_idx], self.tasks[s_idx], is_pred=True)
                target_gt = self.preprocess(gt[t_idx], self.tasks[t_idx], is_pred=False)
                
                # Form canonical task pair (sorted to ensure consistent ordering)
                task_pair = tuple(sorted((s_idx, t_idx)))
                
                # Encode with task-pair conditioning
                map_src = self.encoder(source_pred, s_idx, task_pair)
                map_tgt = self.encoder(target_gt, t_idx, task_pair)
                
                # Compute consistency loss
                total_loss += self.consistency_loss(map_src, map_tgt, feat, reg_weight)
                valid_pairs += 1
        
        return total_loss / max(valid_pairs, 1) if valid_pairs else 0
    
    def preprocess(self, data, task, is_pred=True):
        """Normalize task outputs for consistent processing"""
        if task == 'semantic':
            if is_pred:
                # Use softmax for predictions
                return F.softmax(data, dim=1)
            # Convert GT to one-hot
            mask = (data == -1).float()
            valid_data = data * (1 - mask)
            return F.one_hot(valid_data.long(), self.input_channels[task]).permute(0,3,1,2).float()
        
        elif task == 'depth':
            # Normalize to [0, 1]
            data = data.clone()
            data_max = data.max()
            if data_max > 0:
                return data / data_max
            return data
        
        return data
    
    def consistency_loss(self, src_feat, tgt_feat, backbone_feat, reg_weight):
        """Compute multi-part consistency loss"""
        # Project backbone features
        proj_feat = self.feat_proj(backbone_feat)
        
        # Resize features to match backbone resolution
        src_feat = F.adaptive_avg_pool2d(src_feat, proj_feat.shape[-2:])
        tgt_feat = F.adaptive_avg_pool2d(tgt_feat, proj_feat.shape[-2:])
        
        # Cross-task consistency (cosine similarity)
        cross_loss = 1 - F.cosine_similarity(src_feat, tgt_feat, dim=1).mean()
        
        # Feature alignment regularization
        src_reg = 1 - F.cosine_similarity(src_feat, proj_feat, dim=1).mean()
        tgt_reg = 1 - F.cosine_similarity(tgt_feat, proj_feat, dim=1).mean()
        
        return cross_loss + reg_weight * (src_reg + tgt_reg)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer"""
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(embed_dim, num_features)
        self.beta = nn.Linear(embed_dim, num_features)
        
    def forward(self, x, embedding):
        # Ensure proper dimensions
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
            
        gamma = self.gamma(embedding)
        beta = self.beta(embedding)
        
        # Reshape for broadcasting
        gamma = gamma.view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        
        return x * (1 + torch.sigmoid(gamma)) + beta


class LightweightMapEncoder(nn.Module):
    def __init__(self, input_channels, num_tasks, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        
        # Task embedding layers
        self.task_embeddings = nn.Embedding(num_tasks, embedding_dim)
        
        # Input projections
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for ch in input_channels
        ])
        
        # Shared convolutional layers with FiLM conditioning
        self.conv_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        
        for i in range(num_layers):
            conv = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            film = FiLM(embedding_dim * 2, 64)
            self.conv_layers.append(conv)
            self.film_layers.append(film)
    
    def forward(self, x, input_task_idx, task_pair):
        """Process input with task-specific conditioning
        
        Args:
            x: Input tensor
            input_task_idx: Index of the task for this input
            task_pair: Tuple (task1, task2) for conditioning
        """
        # Project input based on input task
        x = self.input_proj[input_task_idx](x)
        
        # Get task embeddings for both tasks in the pair
        task1_emb = self.task_embeddings(torch.tensor([task_pair[0]], device=x.device))
        task2_emb = self.task_embeddings(torch.tensor([task_pair[1]], device=x.device))
        
        # Concatenate embeddings for conditioning
        task_embed = torch.cat([task1_emb, task2_emb], dim=-1)
        
        # Process through shared layers with FiLM conditioning
        for conv, film in zip(self.conv_layers, self.film_layers):
            x = conv(x)
            x = film(x, task_embed)
        
        return x

 