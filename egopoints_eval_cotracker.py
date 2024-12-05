import sys
sys.path.insert(0, "./co-tracker")
import json
import numpy as np
import random, os, cv2
import torch
from torch.utils.data import DataLoader
from egopoints_dataloader import EgoPointsBenchmark
from cotracker.predictor import CoTrackerOnlinePredictor
import argparse
sys.path.insert(0, "./pips2")
import utils.improc
import utils.misc
import utils.basic


def create_pools(n_pool=1000):
    pools = {}
    pool_names = [
        "old_d_all_avg",
        "new_d_all_avg",
        "median_l2",
        "vis_acc",
        "vis_acc_pos",
        "vis_acc_neg",
        "out_of_view",
        "in_view",
        "RE_ID_acc",
    ]

    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version="np")
    return pools


def test_on_fullseq_cotracker(
    model, d, image_size=(384, 512), max_frames=None
):
    metrics = {}

    seq = str(d["seq"][0])
    print("seq", seq)
    rgb_paths = d["rgb_paths"]
    trajs_g = d["trajs"].cuda().float()  # B,S,N,2
    visibs = d["visibs"].cuda().float()  # B,S,N
    valids = d["valids"].cuda().float()  # B,S,N
    vis_valids = d["vis_valids"].cuda().float() # B,S,N
    out_of_view = d['out_of_view'].cuda().float() # B,S,N
    occ = d["occluded"].cuda().float()

    if max_frames is not None:
        trajs_g = trajs_g[:, :max_frames]
        visibs = visibs[:, :max_frames]
        valids = valids[:, :max_frames]
        vis_valids = vis_valids[:, :max_frames]
    B, T, N, D = trajs_g.shape
    assert D == 2
    assert B == 1
    print("this video is %d frames long" % T)

    # load one to check H,W
    rgb_path0 = rgb_paths[0][0]
    rgb0_bak = cv2.imread(rgb_path0)
    H_bak, W_bak = rgb0_bak.shape[:2]
    H, W = image_size
    sy = H / H_bak
    sx = W / W_bak
    trajs_g[:, :, :, 0] *= sx
    trajs_g[:, :, :, 1] *= sy
    rgb0_bak = cv2.resize(rgb0_bak, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb0_bak = torch.from_numpy(rgb0_bak[:, :, ::-1].copy()).permute(2, 0, 1)  # 3,H,W
    rgb0_bak = rgb0_bak.unsqueeze(0).to(trajs_g.device)  # 1,3,H,W

    queried_coords = trajs_g[:, 0]
    queried_frames = torch.zeros(B, N).long().cuda()

    queries = torch.cat([queried_frames[:, :, None], queried_coords], dim=-1)

    is_first_step = True
    window_frames = []
    for idx in range(T):
        temp_frame = cv2.imread(rgb_paths[idx][0])
        temp_frame = temp_frame[:, :, ::-1]
        H_load, W_load = temp_frame.shape[:2]
        assert H_load == H_bak and W_load == W_bak
        temp_frame = cv2.resize(temp_frame, (W, H), interpolation=cv2.INTER_LINEAR)
        temp_frame = torch.from_numpy(temp_frame).permute(2, 0, 1)
        if idx % model.step == 0 and idx != 0:
            with torch.no_grad():
                video_chunk = torch.tensor(np.stack(window_frames[-model.step*2:])).float().cuda()[None]
                pred_tracks, pred_vis = model(
                    video_chunk,
                    is_first_step=is_first_step,
                    grid_size=10,
                    grid_query_frame=0,
                    queries=queries,
                    add_support_grid=True
                )
            is_first_step = False
        window_frames.append(temp_frame)
    with torch.no_grad():
        video_chunk = torch.tensor(np.stack(window_frames[-model.step*2:])).float().cuda()[None]
        trajs_e, vis_e = model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=10,
            grid_query_frame=0,
            queries=queries,
            add_support_grid=True
        )

    trajs_e = trajs_e[:, :trajs_g.size(1), :N]
    vis_e = vis_e[:, :trajs_g.size(1), :N]
    
    d_avg_thrs, d_avg_star_thrs = [1.0, 2.0, 4.0, 8.0, 16.0], [8.0, 16.0, 24.0]

    ####
    # calculate OOVA
    out_of_view_e = torch.logical_not(torch.logical_and(
        torch.logical_and(trajs_e[:,1:,:,0] >= 0.0, trajs_e[:,1:,:,0] <= W),
        torch.logical_and(trajs_e[:,1:,:,1] >= 0.0, trajs_e[:,1:,:,1] <= H)
    )).float()
    total_out_of_view = torch.sum(out_of_view[:,1:].cpu())
    total_out_of_view_e = 0
    if total_out_of_view != 0:
        total_out_of_view_e = torch.sum(out_of_view[:,1:].cpu() * out_of_view_e.cpu())
        temp_oov = (total_out_of_view_e/total_out_of_view) * 100.0
        metrics["out_of_view"] = (total_out_of_view_e/total_out_of_view) * 100.0
    ###

    # calculate IVA
    in_view_e = torch.logical_and(
        torch.logical_and(trajs_e[:,1:,:,0] >= 0.0, trajs_e[:,1:,:,0] <= W),
        torch.logical_and(trajs_e[:,1:,:,1] >= 0.0, trajs_e[:,1:,:,1] <= H)
    ).float()
    total_in_view = torch.sum(valids[:,1:].cpu() + occ[:,1:].cpu())
    total_in_view_e = 0
    if total_in_view != 0:
        total_in_view_e = torch.sum((valids[:,1:].cpu() + occ[:,1:].cpu()) * in_view_e.cpu())
        temp_iv = (total_in_view_e/total_in_view) * 100.0
        metrics["in_view"] = (total_in_view_e/total_in_view) * 100.0
    ####

    # calculate ReIDd_avg
    if total_out_of_view != 0:
        test = out_of_view[:,1:] * out_of_view_e
        gt_oov_acc_ls, oov_acc_ls = [], []
        for temp_point_idx in range(visibs.size(2)):
            temp_oov_pos = torch.where(test[:, :, temp_point_idx] == 1)
            if len(temp_oov_pos[1]) > 0:
                num_before = torch.max(temp_oov_pos[1]).item() + 1
                oov_acc_ls.append(torch.cat([torch.zeros(num_before), torch.ones(visibs.size(1)-num_before)]).unsqueeze(0))
            else:
                oov_acc_ls.append(torch.zeros(visibs.size(1)).unsqueeze(0))

            temp_oov_pos_gt = torch.where(out_of_view[:, :, temp_point_idx] == 1)
            if len(temp_oov_pos_gt[1]) > 0:
                num_before_gt = torch.max(temp_oov_pos_gt[1]).item() + 1
                gt_oov_acc_ls.append(torch.cat([torch.zeros(num_before_gt), torch.ones(visibs.size(1)-num_before_gt)]).unsqueeze(0))
            else:
                gt_oov_acc_ls.append(torch.zeros(visibs.size(1)).unsqueeze(0))

        oov_acc_valids = torch.cat(oov_acc_ls, dim=0).unsqueeze(0).permute(0,2,1)
        oov_acc_valids_gt = torch.cat(gt_oov_acc_ls, dim=0).unsqueeze(0).permute(0,2,1)
        total_oov_acc = torch.sum(valids[:,1:].cpu() * oov_acc_valids_gt[:,1:].cpu())
        if total_oov_acc != 0:
            re_id_perc_sum = 0.0
            for thr in d_avg_star_thrs:
                re_id_points_within = valids[:,1:].cpu() * oov_acc_valids[:,1:].cpu() * (torch.norm(trajs_e[:,1:]-trajs_g[:,1:], dim=-1) < thr).float().cpu()
                total_oov_acc_precise = torch.sum(re_id_points_within)
                re_id_perc_sum += (total_oov_acc_precise / total_oov_acc) * 100.0
            metrics["RE_ID_acc"] = re_id_perc_sum / len(d_avg_star_thrs)
    ###
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thrs_type, thrs in zip(["new", "old"], [d_avg_star_thrs, d_avg_thrs]):
        d_vis_sum = d_occ_sum = d_sum_all = 0.0
        for thr in thrs:
            # note we exclude timestep0 from this eval
            d_ = (
                torch.norm(trajs_e[:, 1:] / sc_pt - trajs_g[:, 1:] / sc_pt, dim=-1) < thr
            ).float()  # B,S-1,N
            d_all = utils.basic.reduce_masked_mean(d_, valids[:, 1:]).item() * 100.0
            d_sum_all += d_all
            metrics[f"{thrs_type}_d_all_{thr}"] = d_all

        metrics[f"{thrs_type}_d_all_avg"] = d_sum_all / len(thrs)

    # Calculate median l2 for each trajectory
    dists = torch.norm(trajs_e / sc_pt - trajs_g / sc_pt, dim=-1)  # B,S,N
    dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
    valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics["median_l2"] = median_l2.mean().item()

    # get visibility accuracy
    vis_acc_before = (torch.sigmoid(vis_e).round() == visibs).float()
    vis_acc = utils.basic.reduce_masked_mean(vis_acc_before, vis_valids.float())
    temp_vis_pos_valids = (visibs==1)*vis_valids.float()
    vis_acc_pos = utils.basic.reduce_masked_mean(vis_acc_before, temp_vis_pos_valids)
    temp_vis_neg_valids = (visibs==0)*vis_valids.float()
    vis_acc_neg = utils.basic.reduce_masked_mean(vis_acc_before, temp_vis_neg_valids)
    metrics["vis_acc"] = vis_acc.mean().item()
    if torch.any((visibs==1)):
        metrics['vis_acc_pos'] = vis_acc_pos.mean().item()
    if torch.any((visibs==0)):
        metrics['vis_acc_neg'] = vis_acc_neg.mean().item()

    return metrics


def main(
    B=1,  # batchsize
    image_size=(384, 512),  # input resolution
    dataset_location="/media/cibo/DATA/Rhodri/ego_points",
    ckpt_path="/media/deepthought/DATA/Rhodri/cotracker_checkpoints/cotracker2.pth",
    device_ids=[0],
    n_pool=1000,  # how long the running averages should be
):
    device = "cuda:%d" % device_ids[0]

    assert B == 1  # B>1 not implemented here
    assert image_size[0] % 32 == 0
    assert image_size[1] % 32 == 0

    dataset_x = EgoPointsBenchmark(dataset_location=dataset_location)
    dataloader_x = DataLoader(
        dataset_x, batch_size=B, shuffle=False, num_workers=0, drop_last=True
    )
    iterloader_x = iter(dataloader_x)

    print(f"Evaluating model at {ckpt_path}...")
    model = CoTrackerOnlinePredictor(checkpoint=ckpt_path)
    model = model.cuda()
    model.eval()
    
    pools_x = create_pools(n_pool)
    global_step = 0
    max_iters = len(dataset_x)
    print(max_iters)
    while global_step < max_iters:
        global_step += 1
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        try:
            sample = next(iterloader_x)
        except StopIteration:
            iterloader_x = iter(dataloader_x)
            sample = next(iterloader_x)
        with torch.no_grad():
            metrics = test_on_fullseq_cotracker(
                model,
                sample,
                image_size=(384, 512)
            )
            
        for key in list(pools_x.keys()):
            if key in metrics:
                pools_x[key].update([metrics[key]])

        print(
            "step %06d/%d; d_avg %.1f; d_avg* %.1f; ReIDd_avg %.1f; IVA %.1f; OOVA %.1f; OA %.1f; MTE %.1f;"
            % (
                global_step,
                max_iters,
                pools_x["old_d_all_avg"].mean(),
                pools_x["new_d_all_avg"].mean(),
                pools_x["RE_ID_acc"].mean(),
                pools_x["in_view"].mean(),
                pools_x["out_of_view"].mean(),
                pools_x["vis_acc"].mean()*100,
                pools_x["median_l2"].mean()
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder")
    parser.add_argument("--ckpt_path", type=str, help="Path to cotracker checkpoint to evaluate.")
    args = parser.parse_args()
    main(dataset_location=args.dataset_location, ckpt_path=args.ckpt_path)
