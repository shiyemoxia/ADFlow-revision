import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from skimage.measure import label, regionprops
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import load_decoder_arch, load_encoder_arch, positionalencoding2d, activation
from snet_loss import adflow_training_loss, get_logp
from custom_datasets import *
from custom_models import *
from model import load_maca, load_VarianceNet
import torchvision.transforms as transforms
from experiment_utils import CurveRecorder, save_latent_sample
from utils import Score_Observer, rescale, t2np

OUT_DIR = './viz/'

def train_meta_epoch(c, epoch, loader, encoder, decoders, variancenets, optimizer, writer, pool_layers, acas):
    P = c.condition_vec
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    decoders = [decoder.train() for decoder in decoders]
    acas = [ema.train() for ema in acas]
    variancenets = [variancenet.train() for variancenet in variancenets]
    epoch_loss = 0.0
    epoch_nll = 0.0
    epoch_bce = 0.0
    epoch_steps = 0
    with torch.autograd.set_detect_anomaly(c.detect_anomaly):
        for sub_epoch in range(c.sub_epochs):
            start_time = time.time()
            sub_loss = 0.0
            sub_steps = 0
            for i, data in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
                # warm-up learning rate
                lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
                image, _, mask = data
                image = image.to(c.device)
                mask = mask.to(c.device)
                #  encoder prediction
                with torch.no_grad():
                    _ = encoder(image)
                for l, layer in enumerate(pool_layers):
                    decoder = decoders[l]
                    aca = acas[l]
                    #feature_aug = feature_augs[l]
                    variancenet = variancenets[l]
                    e = activation[layer].detach()  # BxCxHxW
                    e = aca(e)
                    B, C, H, W = e.size()
                    S = H * W
                    E = B * S
                    mask_ = F.interpolate(mask, size=(H, W), mode='nearest').squeeze(1)
                    m_r = mask_.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                    p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)  # P is 128
                    # perm = torch.randperm(E).to(c.device)  # BHW

                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    perm = torch.randperm(E, device=c.device)
                    c_p = c_r[perm]
                    e_p = e_r[perm]
                    m_p = m_r[perm]
                    z, log_jac_det = decoder(e_p, [c_p,])

                    var = weights = None
                    if c.distribution == 'adaptive':
                        normal_features = e_r[m_r.squeeze(1) < 0.5]
                        if normal_features.numel() == 0:
                            normal_features = e_r
                        var, weights = variancenet(normal_features)
                    loss, nll, bce = adflow_training_loss(
                        C,
                        z,
                        log_jac_det,
                        m_p,
                        var,
                        weights,
                        distribution=c.distribution,
                        lambda_bce=c.lambda_bce,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_value = float(loss.detach().cpu())
                    nll_value = float(nll.cpu())
                    bce_value = float(bce.cpu())
                    sub_loss += loss_value
                    sub_steps += 1
                    epoch_loss += loss_value
                    epoch_nll += nll_value
                    epoch_bce += bce_value
                    epoch_steps += 1

            mean_train_loss = sub_loss / max(sub_steps, 1)
            writer_step = epoch * c.sub_epochs + sub_epoch
            writer.add_scalar('Loss/train', mean_train_loss, writer_step)
            if c.verbose:
                print("---------------{}s seconds---------------".format(time.time() - start_time))
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss,
                                                                                     lr))
    denominator = max(epoch_steps, 1)
    return {
        'loss': epoch_loss / denominator,
        'nll': epoch_nll / denominator,
        'bce': epoch_bce / denominator,
    }


def test_meta_epoch(c, epoch, loader, encoder, decoders, variancenets, N, pool_layers, acas):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    variancenets = [variancenet.eval() for variancenet in variancenets]
    acas = [ema.eval() for ema in acas]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    latent_z = []
    latent_labels = []
    latent_log_prob = []
    collect_z = c.save_z and epoch == c.meta_epochs - 1
    per_batch_budget = max(1, int(np.ceil(c.z_max_samples / max(len(loader), 1))))
    start = time.time()
    with torch.no_grad():
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            if c.viz:
                image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            # data
            image = image.to(c.device)
            mask_device = mask.to(c.device)
            with torch.no_grad():
                _ = encoder(image)
            for l, layer in enumerate(pool_layers):
                aca = acas[l]
                variancenet = variancenets[l]
                e = activation[layer].detach()  # BxCxHxW
                e = aca(e)
                B, C, H, W = e.size()
                S = H * W
                E = B * S
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)  # P is 128
                # perm = torch.randperm(E).to(c.device)  # BHW
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                m = F.interpolate(mask_device, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                decoder = decoders[l]
                var = weights = None
                if c.distribution == 'adaptive':
                    var, weights = variancenet(e_r)
                FIB = E // N + int(E % N > 0)  # number of fiber batches
                fiber_budget = max(1, int(np.ceil(per_batch_budget / FIB)))
                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * N, (f + 1) * N, device=c.device)
                    else:
                        idx = torch.arange(f * N, E, device=c.device)
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    m_p = m_r[idx] > 0.5  # Nx1
                    z, log_jac_det = decoder(e_p, [c_p,])
                    decoder_log_prob1 = get_logp(
                        C,
                        z,
                        log_jac_det,
                        var,
                        weights,
                        distribution=c.distribution,
                    )
                    log_prob1 = decoder_log_prob1 / C
                    test_loss += float((-log_prob1).sum().cpu())
                    test_count += len(log_prob1)
                    test_dist[l] = test_dist[l] + log_prob1.detach().cpu().tolist()

                    if collect_z and l == c.z_layer:
                        take = min(len(z), fiber_budget)
                        sample_idx = torch.randperm(len(z), device=z.device)[:take]
                        latent_z.append(z[sample_idx].detach().cpu().numpy())
                        latent_labels.append(
                            m_p[sample_idx].reshape(-1).detach().cpu().numpy().astype(np.uint8)
                        )
                        latent_log_prob.append(
                            log_prob1[sample_idx].detach().cpu().numpy()
                        )
    #
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))
    latent_sample = None
    if latent_z:
        latent_sample = {
            'z': np.concatenate(latent_z, axis=0),
            'labels': np.concatenate(latent_labels, axis=0),
            'log_prob': np.concatenate(latent_log_prob, axis=0),
        }
    return (
        height,
        width,
        image_list,
        test_dist,
        gt_label_list,
        gt_mask_list,
        mean_test_loss,
        latent_sample,
    )


def train(c):
    run_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(c.output_dir)
    tensorboard_dir = output_root / 'tensorboard' / c.dataset / c.class_name / c.distribution
    writer = SummaryWriter(str(tensorboard_dir))
    L = c.pool_layers  # number of pooled layers
    #
    img_size = c.crp_size[0]
    resnet_50, pool_layers, pool_dims = load_encoder_arch(c, L)
    for l, p in enumerate(pool_layers):
        print(l, p)
    encoder = resnet_50.to(c.device).eval()
    # NF decoder
    decoders = [load_decoder_arch(c, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.to(c.device) for decoder in decoders]
    MACA = [load_maca(pool_dim) for pool_dim in pool_dims]
    MACA = [ema.to(c.device) for ema in MACA]
    # K value
    variancenets = [load_VarianceNet(pool_dim, c.num_components) for pool_dim in pool_dims]
    variancenets = [variancenet.to(c.device) for variancenet in variancenets]
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
    for l in range(0, L):
        params += list(MACA[l].parameters())
        if c.distribution == 'adaptive':
            params += list(variancenets[l].parameters())
    auc_all = []
    # optimizer
    optimizer = torch.optim.Adam(params, lr=c.lr)
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    # task data
    if c.dataset == 'mvtec':
        test_dataset = MVTecDataset(c, is_train=False)
        anomaly_dataset = MVTecPseudoDataset(c, is_train=True)
    elif c.dataset == 'visa':
        test_dataset = VISA(c, is_train=False)
        anomaly_dataset = VISA(c, is_train=True)
    elif c.dataset == 'visa-new':
        test_dataset = VisADataset(c, is_train=False)
        #train_dataset = MVTecDataset(c, is_train=True)
        anomaly_dataset = VisADataset(c, is_train=True)
    elif c.dataset == 'mvtec-loco':
        test_dataset = MVTecLOCODataset(root=c.data_path,image_size=c.input_size, phase='test',category=c.class_name,to_gpu=False)
        anomaly_dataset = MVTecLOCODataset(root=c.data_path, image_size=c.input_size, phase='train', category=c.class_name,
                                        to_gpu=False)
    elif c.dataset == 'btad':
        test_dataset = BTAD(c, is_train=False)
        anomaly_dataset = BTAD(c, is_train=True)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))

    curve_path = (
        output_root / 'curves' / c.dataset / c.class_name / c.distribution /
        'run_{}.csv'.format(c.run_name)
    )
    curve_recorder = CurveRecorder(
        curve_path,
        {
            'dataset': c.dataset,
            'class_name': c.class_name,
            'distribution': c.distribution,
            'run_name': c.run_name,
            'seed': c.seed,
            'num_components': c.num_components,
            'meta_epochs': c.meta_epochs,
            'sub_epochs': c.sub_epochs,
            'data_path': os.path.abspath(c.data_path),
            'started_at': run_date,
        },
    )

    train_loader = DataLoader(anomaly_dataset, batch_size=c.batch_size, shuffle=True, pin_memory=False, drop_last = True)
    #anomaly_train_loader = DataLoader(anomaly_dataset, batch_size=c.batch_size, shuffle=True, pin_memory=False, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)

    N = 256 # hyperparameter that increases batch size for the decoder model by N
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))
    # stats
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('SEG_AUROC')
    seg_pro_obs = Score_Observer('SEG_AUPRO')
    # additional metrics observers
    det_ap_obs = Score_Observer('DET_AP')
    det_f1_obs = Score_Observer('DET_F1_MAX')
    seg_ap_obs = Score_Observer('SEG_AP')
    seg_f1_obs = Score_Observer('SEG_F1_MAX')
    if c.action_type == 'norm-test':
        c.meta_epochs = 1
    for epoch in range(c.meta_epochs):
        if c.action_type == 'norm-test' and c.checkpoint:
            load_weights_var(MACA, variancenets, decoders, c.checkpoint)  # avoid unnecessary saves)
            train_stats = {'loss': np.nan, 'nll': np.nan, 'bce': np.nan}
        elif c.action_type == 'norm-train':
            print('Train meta epoch: {}'.format(epoch))
            train_stats = train_meta_epoch(
                c,
                epoch,
                train_loader,
                encoder,
                decoders,
                variancenets,
                optimizer,
                writer,
                pool_layers,
                MACA,
            )
        else:
            raise NotImplementedError('{} is not supported action type!'.format(c.action_type))

        (
            height,
            width,
            test_image_list,
            test_dist,
            gt_label_list,
            gt_mask_list,
            mean_test_loss,
            latent_sample,
        ) = test_meta_epoch(
            c,
            epoch,
            test_loader,
            encoder,
            decoders,
            variancenets,
            N,
            pool_layers,
            MACA,
        )
        test_map = [list() for p in pool_layers]
        for l, p in enumerate(pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
            test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[l], width[l])
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                                        size=c.crp_size, mode='bilinear', align_corners=True).squeeze(1).numpy()
        # score aggregation
        score_map = np.zeros_like(test_map[0])
        for l, p in enumerate(pool_layers):
            score_map += test_map[l]
        score_mask = score_map
        # invert probs to anomaly scores
        super_mask = score_mask.max() - score_mask
        # calculate detection AUROC
        score_label = np.max(super_mask, axis=(1, 2))
        gt_label = np.asarray(gt_label_list, dtype=np.bool_)
        det_roc_auc = roc_auc_score(gt_label, score_label)
        save_best_det_weights = det_roc_obs.update(100.0*det_roc_auc, epoch)

        # image-level AP and F1_max
        det_ap = average_precision_score(gt_label, score_label)
        det_precision, det_recall, det_thresholds = precision_recall_curve(gt_label, score_label)
        det_f1_num = 2 * det_precision * det_recall
        det_f1_den = det_precision + det_recall
        det_f1 = np.divide(det_f1_num, det_f1_den, out=np.zeros_like(det_f1_num), where=det_f1_den != 0)
        det_f1_max = float(det_f1.max()) if det_f1.size > 0 else 0.0
        _ = det_ap_obs.update(100.0*det_ap, epoch)
        _ = det_f1_obs.update(100.0*det_f1_max, epoch)

        # calculate segmentation AUROC
        gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool_), axis=1)

        ## resizing supermask
        supermask_resized = F.interpolate(torch.tensor(super_mask).unsqueeze(1), size = gt_mask.shape[1], mode='bilinear', align_corners=True)
        supermask_resized = t2np(supermask_resized)
       # seg_roc_auc = roc_auc_score(gt_mask.flatten(), super_mask.flatten())
        seg_roc_auc = roc_auc_score(gt_mask.flatten(), supermask_resized.flatten())
        save_best_seg_weights = seg_roc_obs.update(100.0*seg_roc_auc, epoch)

        # pixel-level AP and F1_max (use resized supermask consistent with AUROC)
        seg_ap = average_precision_score(gt_mask.flatten(), supermask_resized.flatten())
        seg_precision, seg_recall, seg_thresholds = precision_recall_curve(gt_mask.flatten(), supermask_resized.flatten())
        seg_f1_num = 2 * seg_precision * seg_recall
        seg_f1_den = seg_precision + seg_recall
        seg_f1 = np.divide(seg_f1_num, seg_f1_den, out=np.zeros_like(seg_f1_num), where=seg_f1_den != 0)
        seg_f1_max = float(seg_f1.max()) if seg_f1.size > 0 else 0.0
        _ = seg_ap_obs.update(100.0*seg_ap, epoch)
        _ = seg_f1_obs.update(100.0*seg_f1_max, epoch)

        #if save_best_det_weights and c.action_type != 'norm-test':
            #save_weights_var(acas, variancenets, decoders, c.model, run_date)  # avoid unnecessary saves
        # calculate segmentation AUPRO
        # from https://github.com/YoungGod/DFR:
        seg_pro_auc = 0.0
        if c.pro:  # and (epoch % 4 == 0):  # AUPRO is expensive to compute
            max_step = 1000
            expect_fpr = 0.3  # default 30%
            max_th = super_mask.max()
            min_th = super_mask.min()
            delta = (max_th - min_th) / max_step
            ious_mean = []
            ious_std = []
            pros_mean = []
            pros_std = []
            threds = []
            fprs = []
            binary_score_maps = np.zeros_like(super_mask, dtype=np.bool_)
            for step in range(max_step):
                thred = max_th - step * delta
                # segmentation
                binary_score_maps[super_mask <= thred] = 0
                binary_score_maps[super_mask >  thred] = 1
                pro = []  # per region overlap
                iou = []  # per image iou
                # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
                # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
                for i in range(len(binary_score_maps)):    # for i th image
                    # pro (per region level)
                    label_map = label(gt_mask[i], connectivity=2)
                    props = regionprops(label_map)
                    for prop in props:
                        x_min, y_min, x_max, y_max = prop.bbox 
                        cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                        cropped_mask = prop.filled_image
                        intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                        pro.append(intersection / prop.area)
                    # iou (per image level)
                    intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                    union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
                    if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                        iou.append(intersection / union)

                ious_mean.append(np.array(iou).mean())
                ious_std.append(np.array(iou).std())
                pros_mean.append(np.array(pro).mean())
                pros_std.append(np.array(pro).std())

                gt_masks_neg = ~gt_mask
                fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
                fprs.append(fpr)
                threds.append(thred)

            # as array
            threds = np.array(threds)
            pros_mean = np.array(pros_mean)
            pros_std = np.array(pros_std)
            fprs = np.array(fprs)
            ious_mean = np.array(ious_mean)
            ious_std = np.array(ious_std)
            # best per image iou
            best_miou = ious_mean.max()
            idx = fprs <= expect_fpr
            fprs_selected = fprs[idx]
            fprs_selected = rescale(fprs_selected) 
            pros_mean_selected = pros_mean[idx]    
            seg_pro_auc = auc(fprs_selected, pros_mean_selected)
            _ = seg_pro_obs.update(100.0*seg_pro_auc, epoch)
    #
            auc_all += [[epoch, 100*seg_roc_auc, 100*det_roc_auc, 100*seg_pro_auc]]
        else:
            auc_all += [[epoch, 100*seg_roc_auc, 100*det_roc_auc, 0]]

        curve_recorder.append(
            epoch=epoch,
            train_loss=train_stats['loss'],
            train_nll=train_stats['nll'],
            train_bce=train_stats['bce'],
            test_loss=mean_test_loss,
            det_auroc=100.0 * det_roc_auc,
            seg_auroc=100.0 * seg_roc_auc,
            det_ap=100.0 * det_ap,
            seg_ap=100.0 * seg_ap,
        )

        if latent_sample is not None:
            latent_path = (
                output_root / 'latents' / c.dataset / c.class_name /
                '{}_run{}_layer{}.npz'.format(c.distribution, c.run_name, c.z_layer)
            )
            saved_path = save_latent_sample(
                latent_path,
                latent_sample['z'],
                latent_sample['labels'],
                latent_sample['log_prob'],
                {
                    'dataset': c.dataset,
                    'class_name': c.class_name,
                    'distribution': c.distribution,
                    'run_name': c.run_name,
                    'seed': c.seed,
                    'epoch': epoch,
                    'layer': c.z_layer,
                    'num_components': c.num_components,
                },
                c.z_max_samples,
                c.seed,
            )
            print('Saved latent sample to {}'.format(saved_path))

    save_all(auc_all, c.class_name, run_date)

    # Legacy anomaly-map exports were missing from the provided source tree.
    # Revision plots are produced by plot_revision_experiments.py instead.
    if c.viz:
        print('The legacy --viz exporter was not included in the supplied code; '
              'use plot_revision_experiments.py for revision figures.')

    writer.close()
