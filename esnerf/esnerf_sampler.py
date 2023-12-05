import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from nerfstudio.utils.rich_utils import CONSOLE
import timeit
import numba

@numba.jit()
def accumulate_events(xs, ys, ts, ps, pos_frames, neg_frames):
    assert len(pos_frames) == len(neg_frames)
    
    ts_max = ts.max()
    t_slice = ts_max / len(pos_frames)
    for i in range(len(xs)):  

        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        if t >= ts_max:
            break
        if p > 0:
            pos_frames[int(t // t_slice), y, x] += 1
        else:
            neg_frames[int(t // t_slice), y, x] -= 1

def event_split(file_path: Path, cam_cnt=171, h=540, w=960):  # cam_cnt是图片个数，在下面的Init也要改
    """split event stream and accumulate events"""
    file_path = Path(file_path)
    events = np.load(file_path)
    xs = np.array(events['x'])
    ys = np.array(events['y'])
    ts = np.array(events['t'])
    ps = np.array(events['p'])
    # print("/////////////////////////////////////")
    # print(events['x'])
    # print(events['y'])
    # print(events['t'])
    # print(events['p'])
    # print(xs)
    # print(ys)
    # print(ts)
    # print(ps)
    # print(len(xs),len(ys),len(ts),len(ps))
    # print(cam_cnt - 1)
    # print("/////////////////////////////////////")
    pos_frames = np.zeros((cam_cnt - 1, h, w))
    neg_frames = np.zeros((cam_cnt - 1, h, w))
    accumulate_events(xs, ys, ts, ps, pos_frames, neg_frames)  #这里在报错
    return torch.Tensor(pos_frames), torch.Tensor(neg_frames)

def event_fusion(pos_frames: Tensor, neg_frames, pos_thre, neg_thre, max_winsize=50, device="cuda:0"):

    event_pos_frames = torch.cumsum(pos_frames, dim=0)
    event_neg_frames = torch.cumsum(neg_frames, dim=0)

    print("//////////////////")
    print(pos_frames.size(),neg_frames.size())
    print(pos_thre.size(),neg_thre.size())
    print("//////////////////")

    fusion_frames = pos_frames * pos_thre + neg_frames * neg_thre  
    # 报错 张量大小不同，前面两个是[170,540,960]，后面两个是[1000,260,346]

    event_frames = torch.Tensor().to(device)
    splits = torch.Tensor()
    for i in range(1, len(fusion_frames) + 1): 
        cur_max_winsize = min(i, max_winsize)
    #for i in range(2, len(fusion_frames) + 1): 
    #    cur_max_winsize = min(i - 1, max_winsize - 1)
        winsize = np.random.randint(1, cur_max_winsize + 1)
        event_frame = torch.sum(fusion_frames[i - winsize : i], dim=0, keepdim=True)
        #print(event_frame)
        event_frames = torch.concat((event_frames, event_frame), dim=0)
        splits = torch.concat((splits, Tensor([[i-winsize, i]])), dim=0)
    return event_frames, splits, event_pos_frames, event_neg_frames

def shuffle_array(arr):
    """ Fisher-Yates shuffle"""
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

class ESNerfSampler:
    """Event sampler from accumulation frames"""
    sample_method : Literal["sample_ordered", "sample_1d", "sample_2d", "sample_3d"] = "sample_3d"

    def __init__(self, file_path: Path, pos_thre: Tensor, neg_thre: Tensor, cam_cnt=171, h=540, w=960, 
                 neg_ratio=0.1, max_winsize=1001, batch_size=1024, device="cuda:0"):
        if file_path == "":  # only test
            self.pos_frames, self.neg_frames = 2 * torch.ones((cam_cnt-1, h, w)), -1 * torch.ones((cam_cnt-1, h, w))
        else:
            self.pos_frames, self.neg_frames = event_split(file_path=file_path, cam_cnt=cam_cnt, h=h, w=w)
        #print(self.pos_frames.shape, self.neg_frames.shape)
        self.neg_ratio = neg_ratio
        self.pos_thre = pos_thre
        self.neg_thre = neg_thre
        self.max_winsize = max_winsize
        self.cam_cnt = cam_cnt
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.device = device
        self.pos_frames = self.pos_frames.to(device)
        self.neg_frames = self.neg_frames.to(device)
        self.color_mask = torch.zeros((self.h, self.w, 3), device=device)
        self.count = 0
        if True:
            self.color_mask[0::2, 0::2, 0] = 1 # R
            self.color_mask[0::2, 1::2, 1] = 1 # G
            self.color_mask[1::2, 0::2, 1] = 1 # G
            self.color_mask[1::2, 1::2, 2] = 1 # B
        else:
            self.color_mask[...] = 1
        #print(self.pos_frames.device, self.neg_frames.device)

    def get_selected_frame_2d(self, frame_index, coords_2d):
        splits = self.splits[frame_index * torch.ones(len(coords_2d), dtype=torch.int)]
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords_2d), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords_2d), dim=-1)), dim=0).int()
        color_mask = self.color_mask[coords_2d[:, 0], coords_2d[:, 1]]
        event_frame_selected = self.event_frames[frame_index, coords_2d[:, 0], coords_2d[:, 1]]
        return ray_indices, event_frame_selected, color_mask

    def get_selected_frame(self, coords):
        splits = self.splits[coords[:, 0]]
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords[:, 1:]), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords[:, 1:]), dim=-1)), dim=0).int()
        color_mask = self.color_mask[coords[:, 1], coords[:, 2]]
        event_frame_selected = self.event_frames[coords[:, 0], coords[:, 1], coords[:, 2]]
        event_pos_frame = self.event_pos_frames[coords[:, 0], coords[:, 1], coords[:, 2]]
        return ray_indices, event_frame_selected, color_mask

    def __iter__(self):
        self.event_frames, self.splits, self.event_pos_frames, self.event_neg_frames = \
            event_fusion(pos_frames=self.pos_frames, neg_frames=self.neg_frames, 
                         pos_thre=self.pos_thre, neg_thre=self.neg_thre, 
                         max_winsize=self.max_winsize, device=self.device)
        nonzero_indices = torch.nonzero(self.event_frames)
        self.nonzero_indices_ordered = nonzero_indices
        self.nonzero_indices_3d = nonzero_indices[
            np.random.choice(nonzero_indices.shape[0], size=(nonzero_indices.shape[0],))
        ]
        zero_indices = torch.nonzero(self.event_frames == 0)
        self.zero_indices_ordered = zero_indices
        self.zero_indices_3d = zero_indices[
            np.random.choice(zero_indices.shape[0], size=(zero_indices.shape[0],))
        ]
        self.pos_count_ordered = 0
        self.neg_count_ordered = 0
        self.pos_count_1d = torch.zeros(self.event_frames.shape[0])
        self.neg_count_1d = torch.zeros(self.event_frames.shape[0])
        self.pos_count_2d = 0
        self.neg_count_2d = 0
        self.pos_count_3d = 0
        self.neg_count_3d = 0
        self.frames_order = shuffle_array(list(range(len(self.event_frames))))
        self.frames_order_idx = 0
        return self
    
    def sample_ordered(self, batch_size):
        if self.pos_count_ordered >= len(self.nonzero_indices_ordered):
            self.__iter__()
        pos_size = int(batch_size * (1 - self.neg_ratio))
        neg_size = int(batch_size - pos_size)
        coords = self.nonzero_indices_ordered[self.pos_count_ordered : self.pos_count_ordered + pos_size].to("cpu")
        coords_neg = self.zero_indices_ordered[self.neg_count_ordered : self.neg_count_ordered + neg_size]
        event_frame_selected = self.event_frames[coords[:, 0], coords[:, 1], coords[:, 2]]
        splits = self.splits[coords[:, 0]]
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords[:, 1:]), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords[:, 1:]), dim=-1)), dim=0).int()
        color_mask = self.color_mask[coords[:, 1], coords[:, 2]]
        self.pos_count_ordered += pos_size 
        self.neg_count_ordered += neg_size
        return ray_indices, event_frame_selected, color_mask
    
    def sample_random_1d(self, batch_size):
        pos_size = int(batch_size * (1 - self.neg_ratio))
        neg_size = int(batch_size - pos_size)
        frame_index = np.random.randint(low=0, high=len(self.event_frames))
        coords_2d = torch.nonzero(self.event_frames[frame_index]).to("cpu")
        coords_2d = coords_2d[
            np.random.choice(coords_2d.shape[0], size=(pos_size))
        ]

        # coords_zero_2d包含self.event_frames[frame_index] 中所有值为0的元素的索引
        # 指的是x,y,z轴上分别为0的索引值
        coords_zero_2d = torch.nonzero(self.event_frames[frame_index] == 0).to("cpu")

        print(frame_index)
        print(coords_zero_2d.shape[0])

        coords_zero_2d = coords_zero_2d[
            np.random.choice(coords_zero_2d.shape[0], size=(neg_size))
        ]
        coords_2d = torch.concat((coords_2d, coords_zero_2d), dim=0)

        return self.get_selected_frame_2d(frame_index, coords_2d=coords_2d)

    def sample_random_2d(self, batch_size):
        if self.frames_order_idx >= len(self.frames_order):
            self.__iter__()
        frame_idx = self.frames_order[self.frames_order_idx]
        pos_size = int(batch_size * (1 - self.neg_ratio))
        neg_size = int(batch_size - pos_size)
        if self.pos_count_2d == 0:
            self.nonzero_indices_2d = torch.nonzero(self.event_frames[frame_idx])
            self.nonzero_indices_2d = self.nonzero_indices_2d[
                np.random.choice(self.nonzero_indices_2d.shape[0], size=(self.nonzero_indices_2d.shape[0]))
            ]
            self.zero_indices_2d = torch.nonzero(self.event_frames[frame_idx] == 0)
        coords_2d = self.nonzero_indices_2d[self.pos_count_2d: self.pos_count_2d + pos_size].to("cpu")
        coords_zero_2d = self.zero_indices_2d[self.neg_count_2d: self.neg_count_2d + neg_size].to("cpu")

        coords_2d = torch.concat((coords_2d, coords_zero_2d), dim=0)

        self.pos_count_2d += pos_size
        self.neg_count_2d += neg_size
        if self.pos_count_2d >= len(self.nonzero_indices_2d):
            self.frames_order_idx += 1
            self.pos_count_2d = 0
            self.neg_count_2d = 0
        return self.get_selected_frame_2d(coords_2d=coords_2d)
    
    def sample_random_3d(self, batch_size: int):
        if self.pos_count_3d >= len(self.nonzero_indices_3d):
            self.__iter__()
        pos_size = int(batch_size * (1 - self.neg_ratio))
        neg_size = int(batch_size - pos_size)
        coords_3d = self.nonzero_indices_3d[self.pos_count_3d: self.pos_count_3d + pos_size].to("cpu")
        coords_zero_3d = self.zero_indices_3d[self.neg_count_3d: self.neg_count_3d + neg_size].to("cpu")
        #print(coords_3d.shape, coords_zero_3d.shape)
        coords_3d = torch.concat((coords_3d, coords_zero_3d), dim=0)
        #print(coords_3d)
        event_frame_selected = self.event_frames[coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]]
        splits = self.splits[coords_3d[:, 0]]
        color_mask = self.color_mask[coords_3d[:, 1], coords_3d[:, 2]]
        #print(splits[:, 0][..., None].shape)
        #print(coords[:, 1:].shape)
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords_3d[:, 1:]), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords_3d[:, 1:]), dim=-1)), dim=0).int()
        self.pos_count_3d += pos_size
        self.neg_count_3d += neg_size
        return ray_indices, event_frame_selected, color_mask
    
    def __next__(self):
        self.count += 1
        #if self.count % (5000) == 0:
        #    self.__iter__()
            #if self.neg_ratio < 0.9:
            #    self.neg_ratio += 0.05
        if self.sample_method == "sample_ordered":
            ray_indices, event_frame, color_mask = self.sample_ordered(self.batch_size)
        elif self.sample_method == "sample_2d":
            ray_indices, event_frame, color_mask = self.sample_random_2d(self.batch_size)
        elif self.sample_method == "sample_1d":
            ray_indices, event_frame, color_mask = self.sample_random_1d(self.batch_size)
        else:
            ray_indices, event_frame, color_mask = self.sample_random_3d(self.batch_size)
        #print(ray_indices.shape)
        #print(event_frame.shape)
        event_frame = event_frame[..., None].tile(1, 3) * color_mask
        batch = {"event_frame_selected" : event_frame,
                 "color_mask" : color_mask}

        return ray_indices, batch

if __name__ == "__main__":
    #pos, neg = event_split("/DATA/wyj/EventNeRF/data/lego1/test1/train/events/test_lego1_color.npz")
    #event_sampler = EventSampler("/DATA/wyj/EventNeRF/data/nextnextgen/bottle/train/events/worgb-2022_11_16_15_46_53.npz", pos_thre, neg_thre, cam_cnt, h, w)
    cam_cnt, h, w = 4, 2, 2
    pos_thre = torch.ones((cam_cnt - 1, h, w), device="cuda:0")
    neg_thre = torch.ones((cam_cnt - 1, h, w), device="cuda:0")
    event_sampler = ESNerfSampler("", pos_thre=pos_thre, neg_thre=neg_thre, cam_cnt=cam_cnt, h=h, w=w, max_winsize=2, batch_size=2)
    event_iter = iter(event_sampler)
    batch = next(event_iter)
    batch = next(event_iter)
    #print(pos[0, :10, :10])
    #print(pos[50, 120:130, 170:180])
    #print(neg[0, :10, :10])

