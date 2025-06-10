import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import pdb

def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    return loss

class map_fn(nn.Module):
    def __init__(self, source_channels, target_channels=None):
        super(map_fn, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.filter = filter
        if target_channels is None:
            target_channels = source_channels
        self.pred_encoder_source = nn.ModuleList([self.conv_layer([source_channels[0], filter[0]])])
        self.pred_encoder_target = nn.ModuleList([self.conv_layer([target_channels[0], filter[0]])])
        for i in range(1, len(source_channels)):
            self.pred_encoder_source.append(self.conv_layer([source_channels[i], filter[0]]))
            self.pred_encoder_target.append(self.conv_layer([target_channels[i], filter[0]]))

        # define encoder decoder layers
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
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x, source_index=None, target_index=None, layers=False, feats=False):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * len(self.filter) for _ in range(5))
        for i in range(len(self.filter)):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        # global shared encoder-decoder network
        if source_index is not None:
            x = self.pred_encoder_source[source_index](x)
        else:
            x = self.pred_encoder_target[target_index](x)
        for i in range(len(self.filter)):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        pred = g_maxpool[-1]
        if layers:
            return [g_maxpool[i].mean(-1).mean(-1) for i in range(len(self.filter))]
        if feats:
            return pred, [g_maxpool[i].detach() for i in range(len(self.filter))]
        return pred

    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True)
        )
        return conv_block

    def pre_bn_conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.BatchNorm2d(num_features=channel[0]),
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True)
        )
        return conv_block

class SegFormerMTL(nn.Module):
    def __init__(self, type_='standard', class_nb=13):
        super(SegFormerMTL, self).__init__()
        
        self.type = type_
        self.class_nb = class_nb
        
        # Initialize SegFormer backbone with lightweight configuration
        config = SegformerConfig(
            num_channels=3,
            num_labels=class_nb,
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
        
        # Load pretrained SegFormer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Get feature dimensions from the encoder
        self.feature_dim = 256  # decoder hidden size
        
        # Task-specific heads
        self.semantic_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.class_nb, kernel_size=1)
        )
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        # Cross-task consistency map functions
        self.con_tasks = map_fn(source_channels=[self.class_nb, 1])
        
        # Uncertainty parameters
        self.logsigma_dist = nn.Parameter(torch.FloatTensor([-0.7, -0.7, -0.7, -1.4, -1.4, -1.4]))
        
        # Initialize task-specific heads
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.semantic_head, self.depth_head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size, _, input_h, input_w = x.shape
        
        # FIXED: Use the complete SegFormer model instead of accessing encoder and decoder separately
        # This ensures proper handling of the encoder-decoder connection
        segformer_outputs = self.segformer(x)
        
        # Get the logits from SegFormer (this is the feature representation we want)
        # The logits have shape [batch_size, num_labels, H/4, W/4]
        segformer_logits = segformer_outputs.logits
        
        # Use the logits as features for our task-specific heads
        # First, convert to the expected feature dimension
        if segformer_logits.shape[1] != self.feature_dim:
            # Add a feature projection layer if dimensions don't match
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Conv2d(
                    segformer_logits.shape[1], 
                    self.feature_dim, 
                    kernel_size=1
                ).to(segformer_logits.device)
            features = self.feature_projection(segformer_logits)
        else:
            features = segformer_logits
        
        # Upsample features to match input resolution
        features_upsampled = F.interpolate(
            features, 
            size=(input_h, input_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Task-specific predictions
        semantic_logits = self.semantic_head(features_upsampled)
        depth_pred = self.depth_head(features_upsampled)
        
        # Apply log softmax to semantic predictions
        semantic_pred = F.log_softmax(semantic_logits, dim=1)
        
        # Create feature list for compatibility
        feat = [features, features_upsampled]
        
        return [semantic_pred, depth_pred], self.logsigma_dist, feat

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()

        # semantic loss: depth-wise cross entropy
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)

        # depth loss: l1 norm
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)

        return [loss1, loss2]
    
    def model_fit_task(self, x_pred, x_output, task='semantic'):
        if task == 'semantic':
            loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
        elif task == 'depth':
            # binary mark to mask out undefined pixel space
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)
        return loss

    def model_unsup(self, x_pred_s, x_pred_dt, x_pred_ds, x_pred_nt, x_pred_ns, threshold1=0.95, threshold2=0.1, threshold3=0.1):
        loss1 = self.seg_con(x_pred_s, threshold1)
        loss2 = self.depth_con(x_pred_dt, x_pred_ds, threshold2)
        loss3 = self.normal_con(x_pred_nt, x_pred_ns, threshold3)
        return [loss1, loss2, loss3]

    def seg_con(self, x_pred, x_pred_t=None, threshold=0.95):
        if x_pred_t is None:
            prob, pseudo_labels = F.softmax(x_pred, dim=1).max(1)
            binary_mask = (prob > threshold).type(torch.FloatTensor).cuda()
            loss = F.nll_loss(x_pred, pseudo_labels, reduction='none') * binary_mask
        else:
            prob, pseudo_labels = F.softmax(x_pred_t, dim=1).max(1)
            binary_mask = (prob > threshold).type(torch.FloatTensor).cuda()
            loss = F.nll_loss(x_pred, pseudo_labels, reduction='none') * binary_mask
        return loss.mean()

    def depth_con(self, x_pred, x_pred_s, threshold=0.1):
        binary_mask = ((x_pred.data - x_pred_s.data).abs() < threshold).type(torch.FloatTensor).cuda()
        loss = ((x_pred.data - x_pred_s).abs() * binary_mask).mean()
        return loss

    def normal_con(self, x_pred, x_pred_s, threshold=0.1):
        x_pred_s = x_pred_s / torch.norm(x_pred_s, p=2, dim=1, keepdim=True)
        loss = 1 - (x_pred.data * x_pred_s)
        binary_mask = (loss.data < threshold).type(torch.FloatTensor).cuda()
        loss = (loss * binary_mask).mean()
        return loss

    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda())
                true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda())
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
                union     = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec    = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                            torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                            torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).cuda()
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)

# Alias for backward compatibility
# SegNet = SegFormerMTL