from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        # self.mhead_attention = nn.MultiheadAttention(
        #     embed_dim= 16,
        #     num_heads= 8,
        #     dropout= 0.1,)

        self.mhead_attention_1 = nn.MultiheadAttention(
            embed_dim= 16,
            num_heads= 8,
            dropout= 0.1,)

        self.mhead_attention_2 = nn.MultiheadAttention(
            embed_dim= 32,
            num_heads= 8,
            dropout= 0.1,)

        self.mhead_attention_3 = nn.MultiheadAttention(
            embed_dim= 64,
            num_heads= 8,
            dropout= 0.1,)

        self.mhead_attention_4 = nn.MultiheadAttention(
            embed_dim= 64,
            num_heads= 8,
            dropout= 0.1,)

        self.norm1 = nn.BatchNorm1d(16)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(64)
        self.norm4 = nn.BatchNorm1d(64)




    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        # print("------------------")
        # print(voxel_features)
        # print(voxel_coords)
        # # print(batch_size)
        # print("------------------")

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features, # torch.Size([32000, 4]) 
            indices=voxel_coords.int(), # torch.Size([32000, 4]) 
            spatial_shape=self.sparse_shape, # [41, 1600, 1408]
            batch_size=batch_size # 2
        )
        x = self.conv_input(input_sp_tensor)
        # x_input = x.features.unsqueeze(0)
        # # x_input = x_input.permute(0,2,1)
        # # print(x.features.shape) # torch.Size([32000, 16]) 
        # # print(x.features.device) # cuda:0
        # # print(x.indices.shape) # torch.Size([32000, 4])
        # # print(x.spatial_shape) # [41, 1600, 1408]
        # # print(x.batch_size) # 2
        # attend_features, attend_weights = self.mhead_attention( #官网实现标准注意力
        #     query = x_input, # torch.Size([1, 23905, 16])     
        #     key = x_input, # torch.Size([48, 23905, 16])
        #     value = x_input, # torch.Size([48, 23905, 16])
        #     # key_padding_mask = x.features, # torch.Size([23905, 48])
        # )
        # attend_features = attend_features.squeeze(0) 
        # x = x.replace_feature(attend_features)
        x_conv1 = self.conv1(x)# torch.Size([2, 16, 41, 1600, 1408])
        x_input = x_conv1.features.unsqueeze(0)
        attend_features, attend_weights = self.mhead_attention_1( #官网实现标准注意力
            query = x_input, # torch.Size([1, 23905, 16])     
            key = x_input, # torch.Size([48, 23905, 16])
            value = x_input, # torch.Size([48, 23905, 16])
            # key_padding_mask = x.features, # torch.Size([23905, 48])
        )
        attend_features = self.norm1(attend_features)
        attend_features = attend_features.squeeze(0)
        x_conv1 = x_conv1.replace_feature(attend_features)
        # print(x_conv1.features.shape) # torch.Size([32000, 16]) 
        # print(x_conv1.features.device) # cuda:0
        # print(x_conv1.indices.shape) # torch.Size([32000, 4])
        # print(x_conv1.spatial_shape) # [41, 1600, 1408]
        # print(x_conv1.batch_size) # 2   
        # x_conv1_d = x_conv1.dense()

        x_conv2 = self.conv2(x_conv1)# torch.Size([2, 32, 21, 800, 704])


        x_input = x_conv2.features.unsqueeze(0)
        attend_features, attend_weights = self.mhead_attention_2( #官网实现标准注意力
            query = x_input, # torch.Size([1, 23905, 16])     
            key = x_input, # torch.Size([48, 23905, 16])
            value = x_input, # torch.Size([48, 23905, 16])
            # key_padding_mask = x.features, # torch.Size([23905, 48])
        )
        attend_features = self.norm2(attend_features)
        attend_features = attend_features.squeeze(0)
        x_conv2 = x_conv2.replace_feature(attend_features)


        # print(x_conv2.features.shape) # torch.Size([32000, 32]) 
        # print(x_conv2.features.device) # cuda:0
        # print(x_conv2.indices.shape) # torch.Size([32000, 4])
        # print(x_conv2.spatial_shape) # [21, 800, 704]
        # print(x_conv2.batch_size) # 2   
        # x_conv2_d = x_conv2.dense()
        x_conv3 = self.conv3(x_conv2)# torch.Size([2, 64, 11, 400, 352])

        x_input = x_conv3.features.unsqueeze(0)
        attend_features, attend_weights = self.mhead_attention_3( #官网实现标准注意力
            query = x_input, # torch.Size([1, 23905, 16])     
            key = x_input, # torch.Size([48, 23905, 16])
            value = x_input, # torch.Size([48, 23905, 16])
            # key_padding_mask = x.features, # torch.Size([23905, 48])
        )
        attend_features = self.norm3(attend_features)
        attend_features = attend_features.squeeze(0)
        x_conv3 = x_conv3.replace_feature(attend_features)

        # print(x_conv3.features.shape) # torch.Size([32000, 64]) 
        # print(x_conv3.features.device) # cuda:0
        # print(x_conv3.indices.shape) # torch.Size([32000, 4])
        # print(x_conv3.spatial_shape) # [11, 400, 352]
        # print(x_conv3.batch_size) # 2   
        # x_conv3_d = x_conv3.dense()
        x_conv4 = self.conv4(x_conv3)# torch.Size([2, 64, 5, 200, 176])

        x_input = x_conv4.features.unsqueeze(0)
        attend_features, attend_weights = self.mhead_attention_4( #官网实现标准注意力
            query = x_input, # torch.Size([1, 23905, 16])     
            key = x_input, # torch.Size([48, 23905, 16])
            value = x_input, # torch.Size([48, 23905, 16])
            # key_padding_mask = x.features, # torch.Size([23905, 48])
        )
        attend_features = self.norm4(attend_features)
        attend_features = attend_features.squeeze(0)
        x_conv4 = x_conv4.replace_feature(attend_features)

        # print(x_conv4.features.shape) # torch.Size([32000, 64]) 
        # print(x_conv4.features.device) # cuda:0
        # print(x_conv4.indices.shape) # torch.Size([32000, 4])
        # print(x_conv4.spatial_shape) # [5, 200, 176]
        # print(x_conv4.batch_size) # 2   

        # x_conv4_d = x_conv4.dense()
        # print(x_conv1_d.shape)
        # print(x_conv2_d.shape)
        # print(x_conv3_d.shape)
        # print(x_conv4_d.shape)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        print(x.shape)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
