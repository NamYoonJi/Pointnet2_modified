'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint): 
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def density_based_sample(point, npoint, size = 0.5):   
    """
    밀도 기반 Importance Sampling
    Input:
        point: point cloud data, [N, D] (D >= 3 for [x, y, z])
        npoint: number of samples
        radius: 기준 반지름
    Output:
        point: 샘플링된 포인트의 좌표, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]  

    min_xyz = np.min(xyz, axis=0)  # min값 찾고
    grid_idx = np.floor((xyz - min_xyz) / size ).astype(int)  # size 대로 잘라준 다음에 그리드 인덱스 계산 

    blocks, block_idx = np.unique(grid_idx, axis=0, return_inverse=True)
    block_point_map = {i: [] for i in range(len(blocks))}
    for i, block_id in enumerate(block_idx):
        block_point_map[block_id].append(i)

    # 밀도 기반 k 계산, weight 주기 
    block_weights = np.array([len(indices) for indices in block_point_map.values()]) # 블록 
    block_weights = np.exp(-block_weights) # e^(-weight)로 가중치  줌. 
    # 이 뒤에 정규화를 추가할 수도? 

    # 블록 안에 있는 점들한테 가중치 주기
    weights = np.zeros(N)
    for block_id, indices in block_point_map.items():
        for idx in indices:
            weights[idx] = block_weights[block_id]  

    centroids = np.random.choice(N, size=npoint, replace=False, p=weights)  # 확률 기반 샘플링
    point = point[centroids.astype(np.int32)]  # 샘플링된 포인트 좌표만 반환

    return point

# 전체 데이터의 10%만 random으로 빠르게 뽑고, 뽑힌 점(10%)을 기준으로 나머지 90%를 FPS 적용. 
# def hierarchical_sample(point, npoint):
#     N, D = point.shape
#     xyz = point[:,:3]

#     # Random sampling (Sparse)

#     sparse_point = np.random.choice(N, npoint * 0.1, replace=False) # 10% 만 sparse하게 sampling

#     # FPS (Dense)
#     point = farthest_point_sample(point, npoint - len(point))


#     point = point[centroids.astype(np.int32)]
#     return point 



def load_off_vertices(file_path):
    with open(file_path, 'r') as f:
        # 첫 번째 줄은 'OFF' 문자열이므로 건너뜀
        first_row = f.readline().strip()
        if first_row != 'OFF':
            n_verts, n_faces, _ = map(int, first_row.lstrip('OFF').strip().split())
        
        # 두 번째 줄에서 정점과 면의 개수 읽기
        else:
            n_verts, n_faces, _ = map(int, f.readline().strip().split())
            
        # 정점 좌표 읽기
        vertices = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
    return np.array(vertices, dtype=np.float32)

def pad__points(point_set, target_npoints=1024):
    num_points = point_set.shape[0]
    
    additional_points = np.random.choice(num_points, target_npoints - num_points, replace=True)
    point_set = np.vstack([point_set, point_set[additional_points]])

    return point_set


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        # self.npoints = args.num_point
        self.npoints = 1024
        self.process_data = process_data
        # self.uniform = args.use_uniform_sample
        self.uniform = False
        # self.use_normals = args.use_normals
        self.use_normals = False
        # self.num_category = args.num_category
        self.num_category = 40
        # self.sampling_method = args.sampling_method
        self.sampling_method = args.sampling_method  #여기 바꿈

        shape_ids = {}

        # 각 ids에 리스트 형식으로 ['airplane', 'airplane_0640.off'] 저장
        shape_ids['train'] = [[(line.strip().split(','))[1],(line.strip().split(','))[3]] for line in open('/home/yoon/Desktop/SKKU/DL/DL-TermProject/data/modelnet40_normal_resampled/metadata_modelnet40.csv') if 'train' in line]
        shape_ids['test'] = [[(line.strip().split(','))[1],(line.strip().split(','))[3]] for line in open('/home/yoon/Desktop/SKKU/DL/DL-TermProject/data/modelnet40_normal_resampled/metadata_modelnet40.csv') if 'test' in line]

        self.classes = sorted(list(set([(line.strip().split(','))[1] for line in open('/home/yoon/Desktop/SKKU/DL/DL-TermProject/data/modelnet40_normal_resampled/metadata_modelnet40.csv') if 'test' in line])))

        assert (split == 'train' or split == 'test') # False 뜨면 에러 뽑고 중단.

        self.datapath = [ids for ids in shape_ids[split]]

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        elif self.sampling_method == 'density_based':    #여기 추가함
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_density.dat' % (self.num_category, split, self.npoints))
        else: 
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    cls = self.datapath[index][0]
                    cls = self.classes.index(cls)

                    point_set = load_off_vertices(os.path.join(self.root, self.datapath[index][1]))

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    elif self.sampling_method == 'density_based':    #여기 추가함
                        point_set = density_based_sample(point_set, self.npoints)  ###
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        try:
            if self.process_data:
                # 전처리된 데이터를 메모리에서 바로 로드
                point_set, label = self.list_of_points[index], self.list_of_labels[index]
            else:
                # 전처리되지 않은 데이터를 로드
                label = self.datapath[index][0]
                label = self.classes.index(label)  # 클래스 인덱스로 변환
                
                # OFF 파일 로드
                point_set = load_off_vertices(os.path.join(self.root, self.datapath[index][1]))

                # Farthest Point Sampling 적용 여부
                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

            # 점 클라우드 정규화
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            if not self.use_normals:
                point_set = point_set[:, 0:3]

            # 점 클라우드가 npoints보다 작은 경우 패딩
            if point_set.shape[0] < self.npoints:
                point_set = pad__points(point_set, self.npoints)

        except FileNotFoundError:
            # 파일이 없으면 다음 인덱스로 넘어감
            print(f"File not found: {self.datapath[index][1]}")
            return self._get_item((index + 1) % len(self.datapath))

        return point_set, label


    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    import pandas as pd

    data = ModelNetDataLoader('./data/modelnet40_normal_resampled/', split='train', args=' ')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)