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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, up=False):
        super().__init__()
        
        self.up = up
        
        # Main branch
        layers = []
        if up:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.main_branch = nn.Sequential(*layers)
        
        # Shortcut branch
        shortcut_layers = []
        if up:
            shortcut_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if in_channels != out_channels or stride != 1:
            shortcut_layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1))
            shortcut_layers.append(nn.BatchNorm2d(out_channels))
        
        self.shortcut = nn.Sequential(*shortcut_layers) if shortcut_layers else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_branch(x)
        return F.relu(out + identity)

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Main branch
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut branch with downsampling
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return F.relu(self.main_branch(x) + self.shortcut(x))



# Now let's update the Encoder and Decoder with these specific ResBlock implementations:

class Encoder(nn.Module):
    def __init__(self, img_channels=3,patch_size=16,hidden_dim=192):
        super().__init__()
        stem_kernel = 1
        stem_stride = 1
        stem_padding = 0
        self.patch_size = patch_size
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size**2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size**2))
        
        n_layers = int(np.log2(patch_size))
        

        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=stem_kernel, stride=stem_stride, padding=stem_padding),
        )
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.Sequential(
                ResBlockDown(hidden_dim, hidden_dim),
                ResBlock(hidden_dim, hidden_dim)
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
    def __init__(self,latent_dim=128,patch_size=16,hidden_dim=192):
        super().__init__()
        self.latent_dim = latent_dim
        # Initial dense layer from 128-dim latent to 4x4x256
        self.patch_size = patch_size
        
        n_layers = int(np.log2(patch_size))
        self.proj = nn.Linear(latent_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(ResBlock(hidden_dim, hidden_dim))

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


    def forward(self, x,ids_restore):
        x = self.proj(x)
        
        x = self.inverse_apply_mask(x,ids_restore)
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
    

def mask_to_active_ex(x, mask):
    active_mask = 1 - mask.float()
    b,f,f = mask.shape
    B,C,H,W = x.shape
    scale_h,scale_w = H//f,W//f
    current_active_mask = active_mask.repeat_interleave(scale_h,dim=1).repeat_interleave(scale_w,dim=2)
    x = x*current_active_mask.unsqueeze(1)
    return x


class SparseConv2d(nn.Conv2d):
    def forward(self, x, mask=None):
        x = super().forward(x)
        if mask is not None:    
            x = mask_to_active_ex(x, mask)
        return x

class SparseBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x, mask=None):
        x = super().forward(x)
        if mask is not None:    
            x = mask_to_active_ex(x, mask)
        return x


class ResBlockSparse(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, up=False):
        super().__init__()
        
        self.up = up
        
        # Main branch with sparse layers
        if up:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = None
            
        self.conv1 = SparseConv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = SparseBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SparseConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = SparseBatchNorm2d(out_channels)
        
        # Shortcut branch
        self.shortcut_upsample = None
        self.shortcut_conv = None
        self.shortcut_bn = None
        
        if up:
            self.shortcut_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if in_channels != out_channels or stride != 1:
            self.shortcut_conv = SparseConv2d(in_channels, out_channels, 1, stride=1)
            self.shortcut_bn = SparseBatchNorm2d(out_channels)
        
    def forward(self, x, mask):
        # Apply shortcut
        identity = x
        if self.shortcut_upsample:
            identity = self.shortcut_upsample(identity)
        if self.shortcut_conv:
            identity = self.shortcut_conv(identity, mask)
        if self.shortcut_bn:
            identity = self.shortcut_bn(identity, mask)
            
        # Apply main branch
        out = x
        if self.upsample:
            out = self.upsample(out)
        out = self.conv1(out, mask)
        out = self.bn1(out, mask)
        out = self.relu(out)
        out = self.conv2(out, mask)
        out = self.bn2(out, mask)
        
        return F.relu(out + identity)

class ResBlockDownSparse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Main branch with sparse layers
        self.conv1 = SparseConv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn1 = SparseBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SparseConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = SparseBatchNorm2d(out_channels)
        
        # Shortcut branch with downsampling
        self.shortcut_conv = SparseConv2d(in_channels, out_channels, 1, stride=2)
        self.shortcut_bn = SparseBatchNorm2d(out_channels)
    
    def forward(self, x, mask):
        # Apply main branch
        out = self.conv1(x, mask)
        out = self.bn1(out, mask)
        out = self.relu(out)
        out = self.conv2(out, mask)
        out = self.bn2(out, mask)
        
        # Apply shortcut
        identity = self.shortcut_conv(x, mask)
        identity = self.shortcut_bn(identity, mask)
        
        return F.relu(out + identity)



class SparseEncoderV2(nn.Module):
    def __init__(self, img_channels=3,patch_size=16,hidden_dim=192):
        super().__init__()
        stem_kernel = 1
        stem_stride = 1
        stem_padding = 0
        self.patch_size = patch_size
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size**2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size**2))
        
        n_layers = int(np.log2(patch_size))
        

        self.stem = SparseConv2d(img_channels, hidden_dim, kernel_size=stem_kernel, stride=stem_stride, padding=stem_padding)
        
        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        for i in range(n_layers):
            down_block = ResBlockDownSparse(hidden_dim, hidden_dim)
            res_block = ResBlockSparse(hidden_dim, hidden_dim)
            self.down_blocks.append(down_block)
            self.res_blocks.append(res_block)


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
    
    def forward_backbone(self, x,mask=None):
        if mask is not None:
            f = int(mask.shape[1]**.5)
            mask = mask.reshape(mask.shape[0],f,f)
            
        x = self.stem(x,mask)
        for down_block,res_block in zip(self.down_blocks,self.res_blocks):
            x = down_block(x,mask)
            x = res_block(x,mask)
        return x

    def forward_feature(self, x):

        x = self.forward_backbone(x)
        b,c,h,w = x.shape 
        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        return x

    def forward(self, x, mask_ratio=0.75):
        x_masked, mask, ids_restore = self.apply_mask(x, mask_ratio)
        x = self.forward_backbone(x_masked,mask)
        b,c,h,w = x.shape 

        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        keep_mask = 1-mask

        x = x[keep_mask.bool(),:] # b,h*w,c -> b,h*w_masked,c
        x = x.reshape(b,-1,c) # b,h*w_masked,c -> b,h*w_masked/c
        
        
        return x, mask, ids_restore
    