import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =================================================================================
# Вспомогательные блоки, скопированные из MP-SENet (энкодер, декодеры)
# =================================================================================

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class DenseBlock(nn.Module):
    def __init__(self, channels, kernel_size=(2, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(channels * (i + 1), channels, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(channels, affine=True),
                nn.PReLU(channels)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x_ = self.dense_block[i](skip)
            skip = torch.cat([x_, skip], dim=1)
        return x_

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels))
        self.dense_block = DenseBlock(channels, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels))

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x

class MaskDecoder(nn.Module):
    def __init__(self, channels=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(channels, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.), 
            nn.Conv2d(channels, out_channel, (1, 2))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.sigmoid(x)

class PhaseDecoder(nn.Module):
    def __init__(self, channels=64, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(channels, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

        self.padding = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.phase_conv_r = nn.Conv2d(channels, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(channels, out_channel, (1, 2))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x = self.padding(x) # Применяем паддинг
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return x

# =================================================================================
# Новая реализация Mamba и xLSTM "с нуля"
# =================================================================================

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # --- ИСПРАВЛЕНИЕ НАЧАЛО ---
        # Проекции для B и C теперь отдельные
        self.b_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        # --- ИСПРАВЛЕНИЕ КОНЕЦ ---

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :t]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        
        return self.out_proj(y)

    def ssm(self, x):
        b, t, d_in = x.shape # d_in == d_inner
        d_state = self.d_state

        delta = F.softplus(self.dt_proj(x))
        
        A = -torch.exp(self.A_log.float())
        
        B = self.b_proj(x) # Shape: (b, t, d_state)
        C = self.c_proj(x) # Shape: (b, t, d_state)
        
        # Этот способ вычисления outer product через broadcast работает
        delta_B_x = delta.unsqueeze(-1) * (B.unsqueeze(2) * x.unsqueeze(-1))
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        
        h = torch.zeros(b, d_in, d_state, device=x.device)
        ys = []
        for i in range(t):
            h = deltaA[:, i] * h + delta_B_x[:, i]
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Заменяем матричное умножение на поэлементное с суммой
            y = (C[:, i].unsqueeze(1) * h).sum(dim=-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        return y + x * self.D


class xLSTMBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Проекции для o, i, f гейтов и входа g
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w_i = nn.Linear(d_model, d_model, bias=False)
        self.w_f = nn.Linear(d_model, d_model, bias=False)
        self.w_g = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        b, t, c = x.shape
        
        # Проекции
        o_t = torch.sigmoid(self.w_o(x))
        i_t = torch.exp(self.w_i(x)) # Экспоненциальный гейт
        f_t = torch.sigmoid(self.w_f(x))
        f_t_exp = torch.exp(f_t)
        g_t = torch.tanh(self.w_g(x))

        # Инициализация состояний
        c_s = torch.zeros(b, c, device=x.device)
        c_n = torch.zeros(b, c, device=x.device)
        h = torch.zeros(b, c, device=x.device)
        
        outputs = []
        for i in range(t):
            # Получаем срезы по времени
            ot, it, ft, ftexp, gt = o_t[:,i,:], i_t[:,i,:], f_t[:,i,:], f_t_exp[:,i,:], g_t[:,i,:]
            
            # Нормализация
            c_n = ftexp * c_n + it * gt
            c_s = ftexp * c_s + it
            
            # Вычисление состояния ячейки
            c_t = c_n / c_s
            
            # Вычисление скрытого состояния
            h = ot * c_t
            outputs.append(h)
        
        return self.out_proj(torch.stack(outputs, dim=1))

# =================================================================================
# Основной блок и модель
# =================================================================================

class MambaLSTMProcessingBlock(nn.Module):
    def __init__(self, channels=64):
        super(MambaLSTMProcessingBlock, self).__init__()
        # Используем наши кастомные блоки
        self.time_block = xLSTMBlock(d_model=channels) 
        self.freq_block = MambaBlock(d_model=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, t, f = x.shape
        
        x_res = x
        
        # Обработка по времени (ось T) с помощью xLSTM
        x_time = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_time = self.norm1(x_time)
        x_time = self.time_block(x_time) + x_time # Residual
        
        # Обработка по частоте (ось F) с помощью Mamba
        x_freq = x_time.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_freq = self.norm2(x_freq)
        x_freq = self.freq_block(x_freq) + x_freq # Residual

        out = x_freq.view(b, t, f, c).permute(0, 3, 1, 2)
        return out + x_res # Финальный Residual connection

# Основная модель
class SpeechEnhancementModel(nn.Module):
    def __init__(self, channels=64, num_blocks=4):
        super(SpeechEnhancementModel, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=2, channels=channels)

        self.processing_blocks = nn.ModuleList(
            [MambaLSTMProcessingBlock(channels=channels) for _ in range(num_blocks)]
        )
        
        self.mask_decoder = MaskDecoder(channels=channels, out_channel=1)
        self.phase_decoder = PhaseDecoder(channels=channels, out_channel=1)

    def forward(self, noisy_amp, noisy_pha):
        # noisy_amp, noisy_pha: [B, F, T]
        x = torch.stack((noisy_amp, noisy_pha), dim=1).permute(0, 1, 3, 2) # [B, 2, T, F]
        
        x = self.dense_encoder(x)
        
        for block in self.processing_blocks:
            x = block(x)
        
        denoised_mask = self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)

        input_freq_dim = noisy_amp.shape[1]
        denoised_mask = denoised_mask[:, :input_freq_dim, :]
        denoised_pha = denoised_pha[:, :input_freq_dim, :]
        
        denoised_amp = noisy_amp * denoised_mask

        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha),
             denoised_amp * torch.sin(denoised_pha)), 
            dim=-1
        )

        return denoised_amp, denoised_pha, denoised_com