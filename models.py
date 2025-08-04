#!/usr/bin/env python3
"""
Model architecture components for Hybrid Attention U-Net

This module contains all the neural network components including
attention blocks, encoder/decoder blocks, and the main HybridUNet model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAttentionBlock(nn.Module):
    """Edge Attention Block for early layers focusing on edge information."""
    
    def __init__(self, in_channels):
        super(EdgeAttentionBlock, self).__init__()
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.attention_conv = nn.Conv2d(in_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Canny-like operation
        edge_features = self.relu(self.edge_conv(x))
        
        # Generate attention weights
        attention_weights = self.sigmoid(self.attention_conv(edge_features))
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention Block for deeper layers focusing on spatial features."""
    
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.attention_conv = nn.Conv2d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # Global max along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # Global average along channel dimension
        
        # Concatenate pooling results
        concat_pool = torch.cat([max_pool, avg_pool], dim=1)
        
        # Generate attention weights
        attention_weights = self.sigmoid(self.attention_conv(concat_pool))
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features


class EncoderBlock(nn.Module):
    """Encoder block with optional attention mechanism."""
    
    def __init__(self, in_channels, out_channels, use_edge_attention=False):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.use_edge_attention = use_edge_attention
        
        if use_edge_attention:
            self.attention = EdgeAttentionBlock(out_channels)
        else:
            self.attention = SpatialAttentionBlock()
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Convolution
        conv = self.relu(self.conv(x))
        
        # Apply attention
        conv = self.attention(conv)
        
        # Max pooling
        pooled = self.pool(conv)
        
        return pooled, conv


class DecoderBlock(nn.Module):
    """Decoder block with skip connections."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # After concatenation: out_channels + skip_channels
        self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip_features):
        # Transpose convolution
        up = self.upconv(x)
        
        # Skip connection
        concat = torch.cat([up, skip_features], dim=1)
        
        # Convolution
        conv = self.relu(self.conv(concat))
        
        return conv


class HybridUNet(nn.Module):
    """
    Hybrid U-Net model with Edge and Spatial Attention mechanisms
    for OCT image segmentation of sub-retinal layers.
    """
    
    def __init__(self, input_channels=1, num_classes=4, model_channels=None, use_edge_attention_layers=None):
        super(HybridUNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Default configurations
        if model_channels is None:
            model_channels = [64, 128, 256, 512, 1024]
        if use_edge_attention_layers is None:
            use_edge_attention_layers = [True, True, False, False, False]
        
        # Encoder blocks
        self.enc1 = EncoderBlock(input_channels, model_channels[0], use_edge_attention_layers[0])
        self.enc2 = EncoderBlock(model_channels[0], model_channels[1], use_edge_attention_layers[1])
        self.enc3 = EncoderBlock(model_channels[1], model_channels[2], use_edge_attention_layers[2])
        self.enc4 = EncoderBlock(model_channels[2], model_channels[3], use_edge_attention_layers[3])
        self.enc5 = EncoderBlock(model_channels[3], model_channels[4], use_edge_attention_layers[4])
        
        # Base layer
        self.base_conv = nn.Conv2d(model_channels[4], model_channels[4], 3, padding=1)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_attention = SpatialAttentionBlock()
        
        # Decoder blocks
        self.dec4 = DecoderBlock(model_channels[4], model_channels[4], model_channels[3])
        self.dec3 = DecoderBlock(model_channels[3], model_channels[3], model_channels[2])
        self.dec2 = DecoderBlock(model_channels[2], model_channels[2], model_channels[1])
        self.dec1 = DecoderBlock(model_channels[1], model_channels[1], model_channels[0])
        
        # Final layers
        self.final_upconv = nn.ConvTranspose2d(model_channels[0], model_channels[0], 2, stride=2)
        self.final_conv = nn.Conv2d(model_channels[0] * 2, model_channels[0], 3, padding=1)  # *2 for skip connection
        self.final_relu = nn.ReLU(inplace=True)
        
        # Output layer
        self.output_conv = nn.Conv2d(model_channels[0], num_classes, 1)
        
    def forward(self, x):
        # Encoder path
        enc1, skip1 = self.enc1(x)
        enc2, skip2 = self.enc2(enc1)
        enc3, skip3 = self.enc3(enc2)
        enc4, skip4 = self.enc4(enc3)
        enc5, skip5 = self.enc5(enc4)
        
        # Base layer
        base = self.base_relu(self.base_conv(enc5))
        base = self.base_attention(base)
        
        # Decoder path
        dec4 = self.dec4(base, skip5)
        dec3 = self.dec3(dec4, skip4)
        dec2 = self.dec2(dec3, skip3)
        dec1 = self.dec1(dec2, skip2)
        
        # Final decoder
        final = self.final_upconv(dec1)
        final = torch.cat([final, skip1], dim=1)
        final = self.final_relu(self.final_conv(final))
        
        # Output
        output = self.output_conv(final)
        
        return output
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
