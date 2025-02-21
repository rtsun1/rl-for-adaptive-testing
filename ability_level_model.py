import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shape: int) -> None:
        super(Block, self).__init__()
        shape = [in_channels, shape]
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_emb(t)[:, :, None, None]
        out = x + time_emb
        out = self.ln(x)
        out = self.activation(out)
        out = out + time_emb
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shape: int) -> None:
        super(DownBlock, self).__init__()
        self.block = Block(in_channels, out_channels, shape)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.block(x, t)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shape: int) -> None:
        super(UpBlock, self).__init__()
        self.block = Block(in_channels + out_channels, out_channels, shape)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.block(x, t)
        return out

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, shape) -> None:
        super(MiddleBlock, self).__init__()
        self.block = Block(n_channels, n_channels, shape)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.block(x, t)
        out = self.block(out, t)
        return out

class Upsample(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        out = self.conv(x)
        return out

class Downsample(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        out = self.conv(x)
        return out


# https://nn.labml.ai/diffusion/ddpm/unet.html as reference
class UNet(nn.Module):
    # data in size [num_samples, num_questions, num_data]
    def __init__(self, num_questions: int=244,
                 num_data: int=4, # discrimination, difficulty, response, response time
                 layer_mults = [1, 2, 4], 
                 n_blocks: int=2) -> None:
                 #self, image_channels: int=1,
                 #n_channels: int=32, 
                 #ch_mults = [1, 2, 4], 
                 #time_emb_dim: int=128,
                 #num_classes: int=10,
                 #n_blocks: int=2) -> None:
        super().__init__()

        layer_mults = [num_data * m for m in layer_mults]
        n_resolution = len(layer_mults)
        self.proj = nn.Conv2d(num_questions, layer_mults[0], kernel_size=3, padding=1)
        

        shape = num_data
        # down blocks
        down = []
        out_channels = in_channels = num_data
        for i in range(n_resolution):
            out_channels = layer_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, shape))
                in_channels = out_channels
            
            if i < n_resolution - 1:
                down.append(Downsample(out_channels))
                shape = [shape[0] // 2, shape[1] // 2]
        self.down = nn.ModuleList(down)

        # bottleneck
        self.mid = MiddleBlock(out_channels, shape)

        # up blocks
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolution)):
            out_channels = layer_mults[i]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, shape))
                in_channels = out_channels

            if i > 0:
                up.append(Upsample(out_channels))
                shape = [shape[0] * 2, shape[1] * 2]
        self.up = nn.ModuleList(up)

        # output
        self.norm = nn.LayerNorm([out_channels, shape[0], shape[1]])
        self.activation = nn.SiLU()
        self.final = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: embed time
        x = self.image_proj(x.float())

        history = []
        for down in self.down:
            x = down(x)
            if isinstance(down, DownBlock):
                history.append(x)
        
        x = self.mid(x)

        for up in self.up:
            if isinstance(up, Upsample):
                x = up(x)
            else:
                s = history.pop()
                x = up(torch.cat((x, s), dim=1))
        
        out = self.final(self.activation(self.norm(x)))

        return out