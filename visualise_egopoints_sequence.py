from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_valid_frames(valids):
    valid_frames = []
    for point_idx in range(valids.shape[1]):
        valid_frames.extend(np.where(valids[:, point_idx] == 1.0)[0].tolist())
    return sorted(set(valid_frames))


def load_valid_frames(root, valid_frames):
    img_ls = []
    for valid_frame in valid_frames:
        temp_img = np.array(Image.open(f"{root}/frame_{str(valid_frame+1).zfill(10)}.jpg"))
        img_ls.append(temp_img)
        img_ls.append((np.ones((temp_img.shape[0], 20, 3)) * 255).astype(np.uint8))
    return np.concatenate(img_ls, axis=1), temp_img.shape

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder", default="/media/cibo/DATA/Rhodri/ego_points")
parser.add_argument("--seq_name", type=str, help="EgoPoints sequence name to visualise", default="P01_01_start_15883_end_16103")
args = parser.parse_args()

print(f"Visualising tracks for {args.seq_name}...")

annots = np.load(f"{args.dataset_location}/{args.seq_name}/annot.npz")
trajs_gt = annots["trajs_2d"]
valids = annots["valids"]
valid_frames = get_valid_frames(valids)
concat_frames, img_shape = load_valid_frames(f"{args.dataset_location}/{args.seq_name}/rgbs", valid_frames)

fig, axs = plt.subplots(1, 1, figsize=(24, 8))
axs.axis("off")
axs.imshow(concat_frames)
to_add = img_shape[1] + 20
plot_lines = {}
for frame_num, frame_idx in enumerate(valid_frames):
    for point_idx in range(valids.shape[1]):
        if valids[frame_idx, point_idx] != 1.0:
            continue
        if point_idx not in plot_lines:
            plot_lines[point_idx] = {"coords": []}

        x = trajs_gt[frame_idx, point_idx, 0]
        y = trajs_gt[frame_idx, point_idx, 1]
        plot_lines[point_idx]["coords"].append([(x + (to_add * frame_num)), y])

        if "dynamic_obj_tracks" in annots:
            if annots["dynamic_obj_tracks"][point_idx] == 1:
                plot_lines[point_idx]["linestyle"] = ":"
            else:
                plot_lines[point_idx]["linestyle"] = "-"
        else:
            plot_lines[point_idx]["linestyle"] = "-"

for point_idx in plot_lines:
    temp_coords = np.array(plot_lines[point_idx]["coords"])
    axs.plot(temp_coords[:, 0], temp_coords[:, 1], linestyle=plot_lines[point_idx]["linestyle"], marker="o", label=point_idx)
fig.savefig(f"track_vis_{args.seq_name}.png")
