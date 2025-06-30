import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset
import torch.optim as optim
import os

from tqdm import tqdm 

class FusionDataset(Dataset):
    def __init__(self, dst_files, reori_files, dummy_files):
        self.dst_files = dst_files
        self.reori_files = reori_files
        self.dummy_files = dummy_files

    def split_image_into_patches(self, image, patch_size=(256, 256), overlap=0):
        """
        将大尺寸图像分割为指定大小的patches，并确保每个patch大小一致。
        
        参数:
            image: 输入的大尺寸图像，形状为 [B, C, H, W]
            patch_size: 每个patch的尺寸，默认为 (256, 256)
            overlap: patches之间的重叠量，默认为 0
            
        返回:
            patches: 分割后的图像块，列表形式，每个元素形状为 [B, C, patch_height, patch_width]
            positions: 每个patch在原图中的位置[(y_start, y_end, x_start, x_end)]
        """
        device = image.device  # 获取输入张量所在的设备
        
        batch_size, channels, height, width = image.shape
        patch_height, patch_width = patch_size
        
        # 计算需要的填充量并应用填充
        pad_bottom = (patch_height - height % patch_height) % patch_height
        pad_right = (patch_width - width % patch_width) % patch_width
        padding = (0, pad_right, 0, pad_bottom)  # 左右上下顺序
        padded_image = F.pad(image, padding, "constant", 0).to(device)

        _, _, padded_height, padded_width = padded_image.shape
        
        patches = []
        positions = []

        for y in range(0, padded_height, patch_height - overlap):
            for x in range(0, padded_width, patch_width - overlap):
                y_end = min(y + patch_height, padded_height)
                x_end = min(x + patch_width, padded_width)
                
                patch = padded_image[:, :, y:y_end, x:x_end].to(device)
                
                # 确保每个patch都有相同的尺寸
                if patch.size()[2] < patch_height or patch.size()[3] < patch_width:
                    # 如果不是，则创建一个新tensor并使用padding填充到目标尺寸
                    new_patch = torch.zeros((batch_size, channels, patch_height, patch_width), device=device)
                    new_patch[:, :, :patch.size()[2], :patch.size()[3]] = patch
                    patch = new_patch

                patches.append(patch)
                positions.append((y, y_end, x, x_end))

        return torch.cat(patches), positions

    def merge_predictions(self, predictions, positions, original_shape):
        """
        将所有patch的预测结果合并回原始尺寸的图像。
        
        参数:
            predictions: 预测结果列表，每个元素形状为 [B, 1, patch_height, patch_width]
            positions: 每个patch在原图中的位置列表
            original_shape: 原始图像的形状 [B, C, H, W]
            
        返回:
            merged_image: 合并后的完整图像
        """
        batch_size, channels, height, width = original_shape
        
        # 获取第一个预测结果所在的设备
        device = predictions[0].device
        
        # 初始化合并图像和计数张量
        merged_image = torch.zeros(batch_size, 1, height, width, device=device)
        counts = torch.zeros_like(merged_image, device=device)
        
        for prediction, (y_start, y_end, x_start, x_end) in zip(predictions, positions):

            if prediction.device != device:
                prediction = prediction.to(device)

            # 计算当前patch的实际尺寸
            actual_patch_height = min(y_end - y_start, height - y_start)
            actual_patch_width = min(x_end - x_start, width - x_start)

            if actual_patch_height > 0 and actual_patch_width > 0:
                # 确保只使用与目标区域尺寸相匹配的部分，并且不超出原始图像边界
                end_y = y_start + actual_patch_height
                end_x = x_start + actual_patch_width

                # 确保只使用与目标区域尺寸相匹配的部分
                merged_image[:, :, y_start:end_y, x_start:end_x] += prediction[:, :actual_patch_height, :actual_patch_width]
                counts[:, :, y_start:end_y, x_start:end_x] += 1
        
        # 确保没有除以零的情况
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        
        # 平均化重叠区域
        merged_image /= counts
        
        return merged_image

    def __len__(self):
        return len(self.dst_files)
    
    def __getitem__(self, idx):
        with rasterio.open(self.dst_files[idx]) as dst, \
            rasterio.open(self.reori_files[idx]) as reori, \
            rasterio.open(self.dummy_files[idx]) as dummy:

            dst_tensor = torch.from_numpy(np.nan_to_num(dst.read(), nan=0.0)).float()
            reori_tensor = torch.from_numpy(np.nan_to_num(reori.read(), nan=0.0)).float()
            dummy_tensor = torch.from_numpy(np.nan_to_num(dummy.read(), nan=0.0)).float()

        return dst_tensor, reori_tensor, dummy_tensor

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias,
                     padding_mode='reflect')

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer

class SRLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None):
        super(SRLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x

        out = (self.c5(out))

        return out

    
class ResB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None):
        super(ResB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        shortcut = self.c5(x)

        out = out + shortcut

        return out


class UNetImageFusionSingleChannel(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):  # 修改输入和输出通道数
        super(UNetImageFusionSingleChannel, self).__init__()
        
        # 初始卷积层和Leaky ReLU激活
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(inplace=True)
        )

        self.srlf1 = SRLFB(in_channels=in_channels, out_channels=in_channels)

        # downsample
        self.down = nn.MaxPool2d(2)

        self.srlf2 = SRLFB(in_channels)
        self.res1 = ResB(in_channels)

        self.srlf3 = SRLFB(in_channels)
        self.res2 = ResB(in_channels)

        self.srlf4 = SRLFB(in_channels)
        self.res3 = ResB(in_channels)

        # upsample
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 2, 2)
        self.res4 = ResB(in_channels)
        self.srlf5 = SRLFB(in_channels)

        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 2, 2)
        self.res5 = ResB(in_channels)
        self.srlf6 = SRLFB(in_channels)

        self.up3 = nn.ConvTranspose2d(in_channels, in_channels, 2, 2)
        self.srlf7 = SRLFB(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect')

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='reflect')
        )


    def forward(self, x):
        conv0 = self.conv1(x) 

        srlf_down = self.down(self.srlf1(conv0))

        srlf_res_down_1 = self.down(self.res1(self.srlf2(srlf_down)))

        srlf_res_down_2 = self.down(self.res2(self.srlf3(srlf_res_down_1)))

        srlf_res_up = self.up1(self.res3(self.srlf4(srlf_res_down_2)))

        main = srlf_res_up + srlf_res_down_1

        res_srlf_up_1 = self.up2(self.srlf5(self.res4(main)))

        main = res_srlf_up_1 + srlf_down
        
        res_srlf_up_2 = self.up3(self.srlf6(self.res5(main)))

        main = res_srlf_up_2 + conv0

        srlf_conv = self.conv2(self.srlf7(main))

        main = srlf_conv + x

        main = self.conv3(main)

        return main



def get_sorted_files(folder_path):
    """
    获取指定文件夹下所有.tif文件，并按波段号排序返回。
    
    :param folder_path: 文件夹路径
    :return: 排序后的文件列表
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    # 使用正则表达式提取 B 后面的数字并排序
    files.sort()
    return [os.path.join(folder_path, f) for f in files]

def resample_files(sentinel_files, modis_files, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for file1, file2 in tqdm(zip(sentinel_files, modis_files), 
                             total=min(len(sentinel_files), len(modis_files)), 
                             desc="Resampling Files", unit="file"):
        with rasterio.open(file1) as dataset1, rasterio.open(file2) as dataset2:
            # 获取目标 CRS、变换和尺寸
            dst_crs = dataset1.crs
            dst_transform = dataset1.transform
            dst_width = dataset1.width
            dst_height = dataset1.height

            # 计算变换参数
            transform, width, height = calculate_default_transform(
                dataset2.crs, dst_crs, dataset2.width, dataset2.height,
                *dataset2.bounds,
                dst_width=dst_width,
                dst_height=dst_height
            )

            # 构建输出文件路径
            base_name = os.path.basename(file2)
            output_path = os.path.join(output_folder, base_name)

            # 更新元数据
            kwargs = dataset2.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': dst_width,
                'height': dst_height
            })

            # 创建并写入新文件
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, dataset2.count + 1):
                    reproject(
                        source=rasterio.band(dataset2, i),
                        destination=rasterio.band(dst, i),
                        src_transform=dataset2.transform,
                        src_crs=dataset2.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )

def main(choice, dst_path, ori_path, ori_path2, output_path, patch_size, overlap, EPOCHS, train_num):
    os.makedirs(f'{output_path}/{dst_path}', exist_ok=True)

    dst_files = get_sorted_files(dst_path)
    ori_files = get_sorted_files(ori_path)

    resample_files(dst_files, ori_files,f"re_{ori_path}_2_{dst_path}")

    reori_files = get_sorted_files(f"re_{ori_path}_2_{dst_path}")

    if choice == 'modis sentinel -> landsat':

        ori_files2 = get_sorted_files(ori_path2)

        resample_files(dst_files, ori_files2,f"re_{ori_path2}_2_{dst_path}")

        reori_files2 = get_sorted_files(f"re_{ori_path2}_2_{dst_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if choice == 'modis sentinel -> landsat':
        dataset = FusionDataset(dst_files, reori_files, reori_files2)
    else:
         dataset = FusionDataset(dst_files, reori_files, reori_files)

    # 假设你想用前4个样本作为训练集，其余的作为测试集
    train_indices = list(range(train_num))  # 前4个样本
    test_indices = list(range(train_num, len(dataset)))  # 其余样本

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = UNetImageFusionSingleChannel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # 使用 tqdm 包裹 dataloader，并设置描述信息
        loop = tqdm(enumerate(zip(train_loader, dst_files)), 
                    total=len(train_loader), 
                    desc=f"Epoch [{epoch}/{EPOCHS}]",
                    leave=True)

        for i, ((dst_tensor, reori_tensor, dummy_tensor), file_path) in loop:
            # 现在两张图像尺寸相同，可以沿通道维度拼接了，例如沿通道维度
            if choice == 'modis sentinel -> landsat':
                concatenated_images = torch.cat((reori_tensor, dummy_tensor), dim=1).to(device)
            else:
                #concatenated_images = torch.cat((dst_tensor, reori_tensor), dim=1).to(device)
                concatenated_images = dst_tensor + reori_tensor
            
            # # 步骤 1: 裁剪成 patch
            # patches, positions = dataset.split_image_into_patches(concatenated_images, patch_size, overlap)
            
            # optimizer.zero_grad()
            
            # output_image = model(patches)

            # # 步骤 3: 合并 patch
            # final_image = dataset.merge_predictions(output_image, positions, concatenated_images.shape)

            # 可视化或保存第2个 epoch 的图像
            if epoch % 1 == 0:
                img_single = concatenated_images[0, 0, :, :].detach().cpu().numpy()

                with rasterio.open(file_path) as src:
                    meta = src.meta.copy()

                output_file = f'{output_path}/{file_path}'
                with rasterio.open(output_file, 'w', **meta) as dst:
                    dst.write(img_single, 1)

            # single_image_tensor = final_image.cpu().detach().numpy()[0, 0, :, :]
            # plt.imshow(single_image_tensor)
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()

            # # 计算损失
            # loss = criterion(final_image.to(device), dst_tensor.to(device))
            
            # # 反向传播和优化
            # loss.backward()
            # optimizer.step()
            
            # # 更新进度条上的信息
            # if i % 10 == 0:
            #     loop.set_postfix(loss=loss.item())

    # # 或者仅保存模型的状态字典
    # torch.save(model.state_dict(), 'model_state_dict.pth')


if __name__ == '__main__':
    # landsat modis -> landsat
    # sentinel landsat -> sentinel
    # modis sentinel -> landsat
    choice = 'landsat modis -> landsat'


    ori_path2 = 'Sentinel'

    output_path = 'output'

    patch_size = (1024, 1024)
    overlap = 0

    EPOCHS = 1

    train_num = 6

    for i , j in zip(['0630/Landsat/L0315'], 
                     ['0630/MODIS/MODIS0315']):
        main(choice, i, j, ori_path2, output_path, patch_size, overlap, EPOCHS, train_num)

    
    
