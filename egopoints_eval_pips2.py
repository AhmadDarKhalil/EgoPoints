import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
import utils.misc
import utils.basic
import random, os, cv2
import torch
from fire import Fire
from torch.utils.data import DataLoader
from egopoints_dataloader import EgoPointsBenchmark
import argparse


def create_pools(n_pool=1000):
    pools = {}
    pool_names = [
        'd_1',
        'd_2',
        'd_4',
        'd_8',
        'd_16',
        'new_d_avg',
        'old_d_avg',
        'median_l2',
        'out_of_view',
        'in_view',
        'RE_ID_acc',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np')
    return pools


def test_on_fullseq(model, d, iters=8, S_max=8, image_size=(384,512)):
    metrics = {}

    seq = str(d['seq'][0])
    print('seq', seq)
    trajs_g = d['trajs'].cuda().float() # B,S,N,2
    vis_g = d['visibs'].cuda().float() # B,S,N
    valids = d['valids'].cuda().float() # B,S,N
    vis_valids = d['vis_valids'].cuda().float() # B,S,N
    out_of_view = d['out_of_view'].cuda().float() # B,S,N
    occ = d['occluded'].cuda().float() # B,S,N
    rgb_paths = d['rgb_paths']

    B, S, N, D = trajs_g.shape
    assert(D==2)
    assert(B==1)
    print('this video is %d frames long' % S)

    # load one to check H,W
    rgb_path0 = rgb_paths[0][0]
    rgb0_bak = cv2.imread(rgb_path0)
    H_bak, W_bak = rgb0_bak.shape[:2]
    H, W = image_size
    sy = H/H_bak
    sx = W/W_bak
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    rgb0_bak = cv2.resize(rgb0_bak, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb0_bak = torch.from_numpy(rgb0_bak[:,:,::-1].copy()).permute(2,0,1) # 3,H,W
    rgb0_bak = rgb0_bak.unsqueeze(0).to(trajs_g.device) # 1,3,H,W
        
    # zero-vel init
    trajs_e = trajs_g[:,0].repeat(1,S,1,1)

    cur_frame = 0
    done = False
    feat_init = None
    first_window = True
    while not done:
        end_frame = cur_frame + S_max

        if end_frame > S:
            diff = end_frame-S
            end_frame = end_frame-diff
            cur_frame = max(cur_frame-diff,0)
        print('working on subseq %d:%d' % (cur_frame, end_frame))

        traj_seq = trajs_e[:, cur_frame:end_frame]
        traj_seq_gt = trajs_g[:, cur_frame:end_frame]

        idx_seq = np.arange(cur_frame, end_frame)
        rgb_paths_seq = [rgb_paths[idx][0] for idx in idx_seq]
        rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths_seq]
        rgbs = [rgb[:,:,::-1] for rgb in rgbs] # BGR->RGB
        H_load, W_load = rgbs[0].shape[:2]
        assert(H_load==H_bak and W_load==W_bak)
        rgbs = [cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        rgb_seq = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
        rgb_seq = rgb_seq.unsqueeze(0).to(traj_seq.device) # 1,S,3,H,W
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
        
        preds, _, feat_init, _ = model(
            traj_seq, rgb_seq, iters=iters, feat_init=feat_init,
        )

        trajs_e[:, cur_frame:end_frame] = preds[-1][:, :S_local]
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame] # update the future with new zero-vel
        
        if end_frame >= S:
            done = True
        else:
            cur_frame = cur_frame + S_max - 1

    d_avg_thrs, d_avg_star_thrs = [1.0, 2.0, 4.0, 8.0, 16.0], [8.0, 16.0, 24.0]

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
    eval_idx = torch.where(valids == 1)
    if total_out_of_view != 0:
        test = out_of_view[:,1:] * out_of_view_e
        gt_oov_acc_ls, oov_acc_ls = [], []
        for temp_point_idx in range(vis_g.size(2)):
            temp_oov_pos = torch.where(test[:, :, temp_point_idx] == 1)
            if len(temp_oov_pos[1]) > 0:
                num_before = torch.max(temp_oov_pos[1]).item() + 1
                oov_acc_ls.append(torch.cat([torch.zeros(num_before), torch.ones(vis_g.size(1)-num_before)]).unsqueeze(0))
            else:
                oov_acc_ls.append(torch.zeros(vis_g.size(1)).unsqueeze(0))

            temp_oov_pos_gt = torch.where(out_of_view[:, :, temp_point_idx] == 1)
            if len(temp_oov_pos_gt[1]) > 0:
                num_before_gt = torch.max(temp_oov_pos_gt[1]).item() + 1
                gt_oov_acc_ls.append(torch.cat([torch.zeros(num_before_gt), torch.ones(vis_g.size(1)-num_before_gt)]).unsqueeze(0))
            else:
                gt_oov_acc_ls.append(torch.zeros(vis_g.size(1)).unsqueeze(0))

        oov_acc_valids = torch.cat(oov_acc_ls, dim=0).unsqueeze(0).permute(0,2,1)
        oov_acc_valids_gt = torch.cat(gt_oov_acc_ls, dim=0).unsqueeze(0).permute(0,2,1)
        total_oov_acc = torch.sum(valids[:,1:].cpu() * oov_acc_valids_gt[:,1:].cpu())
        if total_oov_acc != 0:
            re_id_perc_sum = 0.0
            for thr in d_avg_star_thrs:
                total_oov_acc_precise = torch.sum(valids[:,1:].cpu() * oov_acc_valids[:,1:].cpu() * (torch.norm(trajs_e[:,1:]-trajs_g[:,1:], dim=-1) < thr).float().cpu())
                re_id_perc_sum += (total_oov_acc_precise / total_oov_acc) * 100.0
            metrics["RE_ID_acc"] = re_id_perc_sum / len(d_avg_star_thrs)

    ###
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thrs_type, thrs in zip(["new", "old"], [d_avg_star_thrs, d_avg_thrs]):
        d_sum = 0.0
        for thr in thrs:
            # note we exclude timestep0 from this eval
            d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
            d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()*100.0
            d_sum += d_
            metrics['d_%d' % thr] = d_
        d_avg = d_sum / len(thrs)
        metrics[f'{thrs_type}_d_avg'] = d_avg

    # calculate median l2 for each trajectory
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics['median_l2'] = median_l2.mean().item()
        
    return metrics           

    
def main(
    S=128, # seqlen
    stride=8, # spatial stride of the model
    iters=16, # inference steps of the model
    image_size=(384,512),
    dataset_location="/media/cibo/DATA/Rhodri/ego_points",
    ckpt_path="/home/deepthought/Ahmad/pips2/reference_model",
    device_ids=[0],
    n_pool=1000, # how long the running averages should be
):
    device = 'cuda:%d' % device_ids[0]
 
    assert(image_size[0] % 32 == 0)
    assert(image_size[1] % 32 == 0)

    dataset_x = EgoPointsBenchmark(dataset_location=dataset_location)
    dataloader_x = DataLoader(
        dataset_x, batch_size=1, shuffle=False, num_workers=0, drop_last=True
    )
    iterloader_x = iter(dataloader_x)

    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    utils.misc.count_parameters(model)
    _ = saverloader.load(ckpt_path, model.module)
    model.eval()

    pools_x = create_pools(n_pool)
    global_step = 0
    max_iters = len(dataset_x)
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
            metrics = test_on_fullseq(
                model, sample, iters=iters, S_max=S, image_size=image_size
            )
        
        for key in list(pools_x.keys()):
            if key in metrics:
                pools_x[key].update([metrics[key]])

        print(
            "step %06d/%d; d_avg %.1f; d_avg* %.1f; ReIDd_avg %.1f; IVA %.1f; OOVA %.1f; OA --; MTE %.1f;"
            % (
                global_step,
                max_iters,
                pools_x["old_d_avg"].mean(),
                pools_x["new_d_avg"].mean(),
                pools_x["RE_ID_acc"].mean(),
                pools_x["in_view"].mean(),
                pools_x["out_of_view"].mean(),
                pools_x["median_l2"].mean()
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_location", type=str, help="Path to EgoPoints folder", default="/media/cibo/DATA/Rhodri/ego_points")
    parser.add_argument("--ckpt_path", type=str, help="Path to cotracker checkpoint to evaluate.", default="/home/deepthought/Ahmad/pips2/reference_model")
    args = parser.parse_args()
    Fire(main(dataset_location=args.dataset_location, ckpt_path=args.ckpt_path))
