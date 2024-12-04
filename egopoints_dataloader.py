import torch
import numpy as np
import glob


class EgoPointsBenchmark(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_location="/media/cibo/DATA/Rhodri/ego_points"
    ):
        self.annotation_paths = sorted(glob.glob(f"{dataset_location}/*/annot.npz"))
        print(len(self.annotation_paths))
        self.rgb_paths = [sorted(glob.glob(f"{temp_path.split('/annot.npz')[0]}/rgbs/*.jpg")) for temp_path in self.annotation_paths]
        print(len(self.rgb_paths))
 
    def __getitem__(self, index):
        rgb_paths = self.rgb_paths[index]
        seq = self.annotation_paths[index].split("/")[-2]
        annotations = np.load(self.annotation_paths[index])
        trajs = annotations["trajs_2d"].astype(np.float32)
        valids = annotations["valids"].astype(np.float32)
        visibs = annotations["visibs"].astype(np.float32)
        vis_valids = annotations["vis_valids"].astype(np.float32)
        out_of_view = annotations["out_of_view"].astype(np.float32)
        occluded = annotations["occluded"].astype(np.float32)
        sample = {
            'seq': seq,
            'rgb_paths': rgb_paths,
            'trajs': trajs,
            'visibs': visibs,
            'valids': valids,
            'vis_valids': vis_valids,
            'out_of_view': out_of_view,
            'occluded': occluded
        }
        return sample

    def __len__(self):
        return len(self.rgb_paths)
