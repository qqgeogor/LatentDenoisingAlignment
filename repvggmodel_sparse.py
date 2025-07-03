import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add a proper reshape layer
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert kernel_size == 3
        assert padding == 1

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, groups=groups, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, inputs,mask=None):
        if hasattr(self, 'rbr_reparam'):
            return F.relu(self.rbr_reparam(inputs)),mask

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        x = F.relu(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

        if mask is not None:
            active_mask = 1 - mask.float()
            b,l = active_mask.shape
            f = int(l**.5)
            active_mask = active_mask.reshape(b,f,f)
            
            B,C,H,W = x.shape
            scale_h,scale_w = H//f,W//f
            current_active_mask = active_mask.repeat_interleave(scale_h,dim=1).repeat_interleave(scale_w,dim=2)
            x = x*current_active_mask.unsqueeze(1)


        return  x,mask

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()

class RepVGGBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.repvgg_block = RepVGGBlock(in_channels, out_channels, deploy=deploy)
        
    def forward(self, x):
        x = self.upsample(x)
        return self.repvgg_block(x)

class RepVGGBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                      padding=1, groups=1, bias=True)
        else:
            # No identity branch for downsampling since stride=2
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, inputs,mask=None):
        if hasattr(self, 'rbr_reparam'):
            return F.relu(self.rbr_reparam(inputs)),mask

        x = F.relu(self.rbr_dense(inputs) + self.rbr_1x1(inputs))

        if mask is not None:
            active_mask = 1 - mask.float()
            b,l = active_mask.shape
            f = int(l**.5)
            active_mask = active_mask.reshape(b,f,f)
            
            B,C,H,W = x.shape
            scale_h,scale_w = H//f,W//f
            current_active_mask = active_mask.repeat_interleave(scale_h,dim=1).repeat_interleave(scale_w,dim=2)
            x = x*current_active_mask.unsqueeze(1)
        return x,mask

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()




# Custom Sequential class that can handle (x, mask) tuples
class MaskedSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
    
    def forward(self, x, mask=None):
        for module in self.modules_list:
            if hasattr(module, 'forward') and 'mask' in module.forward.__code__.co_varnames:
                x, mask = module(x, mask)
            else:
                x = module(x)
        return x, mask



# Now let's update the Encoder and Decoder with RepVGG implementations:

class Encoder(nn.Module):
    def __init__(self, img_channels=3, patch_size=16, hidden_dim=192):
        super().__init__()
        stem_kernel = 1
        stem_stride = 1
        stem_padding = 0
        self.patch_size = patch_size
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size**2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size**2))
        
        n_layers = int(np.log2(patch_size))
        


        all_hidden_dims = [hidden_dim] 
        for i in range(n_layers-1):
            all_hidden_dims.append(all_hidden_dims[-1]//2)
        all_hidden_dims = all_hidden_dims[::-1]

        print('all_hidden_dims', all_hidden_dims)
        print('n_layers', n_layers)

        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, all_hidden_dims[0]//2, kernel_size=stem_kernel, stride=stem_stride, padding=stem_padding),
        )
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            current_hidden_dim = all_hidden_dims[i]
            print('current_hidden_dim', current_hidden_dim)
            layer = MaskedSequential(
                RepVGGBlockDown(current_hidden_dim//2, current_hidden_dim),
                RepVGGBlock(current_hidden_dim, current_hidden_dim),
                RepVGGBlock(current_hidden_dim, current_hidden_dim),
                RepVGGBlock(current_hidden_dim, current_hidden_dim)
            )
            self.layers.append(layer)
        self.initialize_weights()

    def apply_mask(self, x, mask_ratio):
        # b,c,h,w -> b,c,h*w -> b,h*w,c
        x = self.patchify(x)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # b,h*w,c -> b,h*w+1,c
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # b,h*w+1,c -> b,h*w,c
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x[:, 1:, :]

        # b,h*w,c -> b,c,h,w
        x = self.unpatchify(x)

        return x, mask, ids_restore

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_backbone(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_feature(self, x):
        x = self.forward_backbone(x)
        b,c,h,w = x.shape 
        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        return x

    def forward(self, x, mask_ratio=0.75):
        x_masked, mask, ids_restore = self.apply_mask(x, mask_ratio)
        x = self.forward_backbone(x_masked)
        b,c,h,w = x.shape 

        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        keep_mask = 1-mask

        x = x[keep_mask.bool(),:] # b,h*w,c -> b,h*w_masked,c
        x = x.reshape(b,-1,c) # b,h*w_masked,c -> b,h*w_masked/c
        
        return x, mask, ids_restore

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, patch_size=16, hidden_dim=192):
        super().__init__()
        self.latent_dim = latent_dim
        # Initial dense layer from 128-dim latent to 4x4x256
        self.patch_size = patch_size
        
        n_layers = int(np.log2(patch_size))
        self.proj = nn.Linear(latent_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.Sequential(  
                RepVGGBlock(hidden_dim, hidden_dim),
                RepVGGBlock(hidden_dim, hidden_dim),
            )
            self.layers.append(layer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.decoder_masked_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder_masked_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def inverse_apply_mask(self, x, ids_restore):
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # b,h*w,c -> b,h*w+1,c
        mask_tokens = self.decoder_masked_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # b,h*w+1,c -> b,h*w,c
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x[:, 1:, :]
        h,w = int(x.shape[1]**.5),int(x.shape[1]**.5)
        x = x.permute(0,2,1)
        x = x.reshape(x.shape[0],-1,h,w)
        return x

    def forward(self, x, ids_restore):
        x = self.proj(x)
        
        x = self.inverse_apply_mask(x, ids_restore)
        for layer in self.layers:
            x = layer(x)

        b,c,h,w = x.shape
        x = x.reshape(b,c,h*w).permute(0,2,1)

        return x 


class SparseEncoder(Encoder):
    def __init__(self, img_channels=3,patch_size=16,hidden_dim=192,sparse=False):
        super().__init__(img_channels,patch_size,hidden_dim)
        self.sparse = sparse
        
    def forward_backbone(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x,_ = layer(x)
        return x

    # x: b,c,h,w
    def forward_backbone_sparse(self, x,mask):

        x = self.stem(x)

        for layer in self.layers:
            x,_ = layer(x,mask)


        return x




    def forward_feature(self, x):
        x = self.forward_backbone(x)
        b,c,h,w = x.shape 
        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        return x

    def forward(self, x, mask_ratio=0.75):
        x_masked, mask, ids_restore = self.apply_mask(x, mask_ratio)
        if self.sparse:
            x = self.forward_backbone_sparse(x_masked,mask)
        else:
            x = self.forward_backbone(x_masked)
        b,c,h,w = x.shape 

        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        keep_mask = 1-mask

        x = x[keep_mask.bool(),:] # b,h*w,c -> b,h*w_masked,c
        x = x.reshape(b,-1,c) # b,h*w_masked,c -> b,h*w_masked/c
        
        
        return x, mask, ids_restore
    

def mask_to_active_ex(x, mask):
    active_mask = 1 - mask.float()
    b,f,f = mask.shape
    B,C,H,W = x.shape
    scale_h,scale_w = H//f,W//f
    current_active_mask = active_mask.repeat_interleave(scale_h,dim=1).repeat_interleave(scale_w,dim=2)
    x = x*current_active_mask.unsqueeze(1)
    return x



class SparseEncoderV2(Encoder):
    def __init__(self, img_channels=3,patch_size=16,hidden_dim=192,sparse=False):
        super().__init__(img_channels,patch_size,hidden_dim)
        self.sparse = sparse
        
    def forward_backbone(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x

    # x: b,c,h,w
    def forward_backbone_sparse(self, x,mask):

        active_mask = 1 - mask.float()
        b,l = active_mask.shape
        f = int(l**.5)
        active_mask = active_mask.reshape(b,f,f)
        
        # input shape
        B,C,H,W = x.shape
        

        scale_h,scale_w = H//f,W//f
        current_active_mask = active_mask.repeat_interleave(scale_h,dim=1).repeat_interleave(scale_w,dim=2)
        

        x = self.stem(x)
        x = x*current_active_mask.unsqueeze(1)


        for layer in self.layers:
            x = layer(x)
            B,C,H,W = x.shape
            scale_h,scale_w = H//f,W//f
            current_active_mask = active_mask.repeat_interleave(scale_h,dim=1).repeat_interleave(scale_w,dim=2)
            x = x*current_active_mask.unsqueeze(1)

        return x




    def forward_feature(self, x):
        x = self.forward_backbone(x)
        b,c,h,w = x.shape 
        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        return x

    def forward(self, x, mask_ratio=0.75):
        x_masked, mask, ids_restore = self.apply_mask(x, mask_ratio)
        if self.sparse:
            x = self.forward_backbone_sparse(x_masked,mask)
        else:
            x = self.forward_backbone(x_masked)
        b,c,h,w = x.shape 

        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        keep_mask = 1-mask

        x = x[keep_mask.bool(),:] # b,h*w,c -> b,h*w_masked,c
        x = x.reshape(b,-1,c) # b,h*w_masked,c -> b,h*w_masked/c
        
        
        return x, mask, ids_restore
    