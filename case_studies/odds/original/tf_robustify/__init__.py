# Copyright 2022 The Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

import os
import numpy as np
import tqdm
import math
import scipy.stats
from absl import logging
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import itertools as itt
from types import SimpleNamespace


logging.set_verbosity(logging.INFO)


def collect_statistics(x_train, y_train, x_ph=None, sess=None,
    latent_and_logits_fn_th=None, latent_x_tensor=None, logits_tensor=None,
    nb_classes=None, weights=None, cuda=True, targeted=False, noise_eps=8e-3,
    noise_eps_detect=None, num_noise_samples=256, batch_size=256,
    pgd_eps=8/255, pgd_lr=1/4, pgd_iters=10, clip_min=-1., clip_max=1.,
    p_ratio_cutoff=20., save_alignments_dir=None, load_alignments_dir=None,
    debug_dict=None, debug=False, clip_alignments=True, pgd_train=None,
    fit_classifier=False, just_detect=False):
    assert len(x_train) == len(y_train)
    if pgd_train is not None:
        assert len(pgd_train) == len(x_train)

    if x_ph is not None:
        import tensorflow as tf
        backend = 'tf'
        assert sess is not None
        assert latent_and_logits_fn_th is None
        assert latent_x_tensor is not None
        assert logits_tensor is not None
        assert nb_classes is not None
        assert weights is not None
    else:
        import torch as th
        backend = 'th'
        assert x_ph is None
        assert sess is None
        assert latent_and_logits_fn_th is not None
        assert latent_x_tensor is None
        assert logits_tensor is None
        assert nb_classes is not None
        assert weights is not None
        cuda = th.cuda.is_available() and cuda

        def latent_fn_th(x):
            return to_np(latent_and_logits_fn_th(to_th(x))[0])

        def logits_fn_th(x):
            return latent_and_logits_fn_th(x)[1]

        def to_th(x, dtype=np.float32):
            x = th.from_numpy(x.astype(dtype))
            if cuda:
                x = x.cuda()
            return x

        def to_np(x):
            return x.detach().cpu().numpy()

    if debug:
        logging.set_verbosity(logging.DEBUG)

    try:
        len(noise_eps)
        if isinstance(noise_eps, str):
            raise TypeError()
    except TypeError:
        noise_eps = [noise_eps]

    if noise_eps_detect is None:
        noise_eps_detect = noise_eps

    try:
        len(noise_eps_detect)
        if isinstance(noise_eps_detect, str):
            raise TypeError()
    except TypeError:
        noise_eps_detect = [noise_eps_detect]

    noise_eps_all = set(noise_eps + noise_eps_detect)

    pgd_lr = pgd_eps * pgd_lr
    n_batches = math.ceil(x_train.shape[0] / batch_size)

    if len(y_train.shape) == 2:
        y_train = y_train.argmax(-1)

    if backend == 'tf':
        y_ph = tf.placeholder(tf.int64, [None])
        loss_tensor = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tensor, labels=y_ph))
        pgd_gradients = tf.gradients(loss_tensor, x_ph)[0]
        preds_tensor = tf.arg_max(logits_tensor, -1)
    else:
        loss_fn = th.nn.CrossEntropyLoss(reduce='sum')
        if cuda:
            loss_fn = loss_fn.cuda()

    def get_noise_samples(x, num_samples, noise_eps, clip=False):
        if isinstance(noise_eps, float):
            kind = 'u'
            eps = noise_eps
        else:
            kind, eps = noise_eps[:1], float(noise_eps[1:])

        if isinstance(x, np.ndarray):
            if kind == 'u':
                noise = np.random.uniform(-1., 1., size=(num_samples,) + x.shape[1:])
            elif kind == 'n':
                noise = np.random.normal(0., 1., size=(num_samples,) + x.shape[1:])
            elif kind == 's':
                noise = np.random.uniform(-1., 1., size=(num_samples,) + x.shape[1:])
                noise = np.sign(noise)
            x_noisy = x + noise * eps
            if clip:
                x_noisy = x_noisy.clip(clip_min, clip_max)
        elif backend == 'tf':
            shape = (num_samples,) + tuple(s.value for s in x.shape[1:])
            if kind == 'u':
                noise = tf.random_uniform(shape=shape, minval=-1., maxval=1.)
            elif kind == 'n':
                noise = tf.random_normal(shape=shape, mean=0., stddev=1.)
            elif kind == 's':
                noise = tf.random_uniform(shape=shape, minval=-1., maxval=1.)
                noise = tf.sign(noise)
            x_noisy = x + noise * eps
            if clip:
                x_noisy = tf.clip_by_value(x_noisy, clip_min, clip_max)
        elif backend == 'th':
            if kind == 'u':
                noise = x.new_zeros((num_samples,) + x.shape[1:]).uniform_(-1., 1.)
            elif kind == 'n':
                noise = x.new_zeros((num_samples,) + x.shape[1:]).normal_(0., 1.)
            elif kind == 's':
                noise = x.new_zeros((num_samples,) + x.shape[1:]).uniform_(-1., 1.)
                noise.sign_()
            x_noisy = x + noise * eps
            if clip:
                x_noisy.clamp_(clip_min, clip_max)
        return x_noisy

    def attack_pgd(x, x_pred, targeted=False):
        x_pgd = get_noise_samples(x, x.shape[0], pgd_eps, clip=True)

        for _ in range(pgd_iters):
            if backend == 'tf':
                x_grads = sess.run(pgd_gradients, {x_ph: x_pgd, y_ph: x_pred})
            else:
                x_th = to_th(x_pgd).requires_grad_(True)
                x_grads = to_np(th.autograd.grad(loss_fn(logits_fn_th(x_th), to_th(x_pred, np.long)), [x_th])[0])

            x_pgd += pgd_lr * np.sign(x_grads) * (-2. * (targeted - 1/2))
            x_pgd = x_pgd.clip(x - pgd_eps, x + pgd_eps)
            x_pgd = x_pgd.clip(clip_min, clip_max)
            if debug:
                break
        return x_pgd

    def get_latent_and_pred(x):
        if backend == 'tf':
            return sess.run([latent_x_tensor, preds_tensor], {x_ph: x})
        else:
            l, p = map(to_np, latent_and_logits_fn_th(to_th(x)))
            return l, p.argmax(-1)

    x_preds_clean = []
    x_train_pgd = []
    x_preds_pgd = []
    latent_clean = []
    latent_pgd = []

    if not load_alignments_dir:
        for b in tqdm.trange(n_batches, desc='creating adversarial samples'):
            x_batch = x_train[b*batch_size:(b+1)*batch_size]
            lc, pc = get_latent_and_pred(x_batch)
            x_preds_clean.append(pc)
            latent_clean.append(lc)

            if not just_detect:
                if pgd_train is not None:
                    x_pgd = pgd_train[b*batch_size:(b+1)*batch_size]
                else:
                    if targeted:
                        x_pgd = np.stack([attack_pgd(x_batch, np.ones_like(pc) * i, targeted=True) for i in range(nb_classes)], 1)
                    else:
                        x_pgd = attack_pgd(x_batch, pc, targeted=False)
                x_train_pgd.append(x_pgd)

                if targeted:
                    pps, lps = [], []
                    for i in range(x_pgd.shape[1]):
                        lp, pp = get_latent_and_pred(x_pgd[:, i])
                        pps.append(pp)
                        lps.append(lp)
                    x_preds_pgd.append(np.stack(pps, 1))
                    latent_pgd.append(np.stack(lps, 1))
                else:
                    lp, pp = get_latent_and_pred(x_pgd)
                    x_preds_pgd.append(pp)
                    latent_pgd.append(lp)

        x_preds_clean, latent_clean = map(np.concatenate, (x_preds_clean, latent_clean))
        if not just_detect:
            x_train_pgd, x_preds_pgd, latent_pgd = map(np.concatenate, (x_train_pgd, x_preds_pgd, latent_pgd))

        valid_idcs = []
        if not just_detect:
            for i, (pc, pp, y) in enumerate(zip(x_preds_clean, x_preds_pgd, y_train)):
                if y == pc and pc != pp:
                # if y == pc:
                    valid_idcs.append(i)
        else:
            valid_idcs = list(range(len(x_preds_clean)))

        logging.info('valid idcs ratio: {}'.format(len(valid_idcs) / len(y_train)))
        if targeted:
            for i, xpp in enumerate(x_preds_pgd.T):
                logging.info('pgd success class {}: {}'.format(i, (xpp == i).mean()))

        x_train, y_train, x_preds_clean, latent_clean = (a[valid_idcs] for a in (x_train, y_train, x_preds_clean, latent_clean))
        if not just_detect:
            x_train_pgd, x_preds_pgd, latent_pgd = (a[valid_idcs] for a in (x_train_pgd, x_preds_pgd, latent_pgd))

    if backend == 'tf':
        weights = tf.transpose(weights, (1, 0))
        weights_np = sess.run(weights)
    else:
        weights_np = weights.cpu().numpy()
    big_memory = weights.shape[0] > 20
    logging.info('BIG MEMORY: {}'.format(big_memory))
    if not big_memory:
        wdiffs = weights[None, :, :] - weights[:, None, :]
        wdiffs_np = weights_np[None, :, :] - weights_np[:, None, :]

    if backend == 'tf':
        # lat_ph = tf.placeholder(tf.float32, [weights.shape[-1]])
        # pred_ph = tf.placeholder(tf.int64)
        # if big_memory:
            # wdiffs_relevant = weights[pred_ph, None] - weights
        # else:
            # wdiffs_relevant = wdiffs[:, pred_ph]
        # lat_diff_tensor = lat_ph[None] - latent_x_tensor
        # alignments_tensor = tf.matmul(lat_diff_tensor, wdiffs_relevant, transpose_b=True)

        # def _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps):
            # x_noisy = get_noise_samples(x[None], num_noise_samples, noise_eps=neps, clip=clip_alignments)
            # return sess.run(alignments_tensor, {x_ph: x_noisy, lat_ph: lat, pred_ph: pred})[:, idx_wo_pc]
        lat_ph = tf.placeholder(tf.float32, [weights.shape[-1]])
        wdiffs_relevant_ph = tf.placeholder(tf.float32, [weights.shape[-1], nb_classes])
        lat_diff_tensor = lat_ph[None] - latent_x_tensor
        alignments_tensor = tf.matmul(lat_diff_tensor, wdiffs_relevant_ph)

        def _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps):
            if big_memory:
                wdiffs_relevant = weights_np[pred, None] - weights_np
            else:
                wdiffs_relevant = wdiffs_np[:, pred]
            x_noisy = get_noise_samples(x[None], num_noise_samples, noise_eps=neps, clip=clip_alignments)
            # return sess.run(alignments_tensor, {x_ph: x_noisy, lat_ph: lat, wdiffs_relevant_ph:wdiffs_relevant.T})[:, idx_wo_pc]
            lat_x = sess.run(latent_x_tensor, {x_ph: x_noisy})
            lat_diffs = lat[None] - lat_x
            return np.matmul(lat_diffs, wdiffs_relevant.T)[:, idx_wo_pc]

    else:

        def _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps):
            x, lat = map(to_th, (x, lat))
            if big_memory:
                wdiffs_relevant = weights[pred, None] - weights
            else:
                wdiffs_relevant = wdiffs[:, pred]
            x_noisy = get_noise_samples(x[None], num_noise_samples, noise_eps=neps, clip=clip_alignments)
            lat_noisy, _ = latent_and_logits_fn_th(x_noisy)
            lat_diffs = lat[None] - lat_noisy
            return to_np(th.matmul(lat_diffs, wdiffs_relevant.transpose(1, 0)))[:, idx_wo_pc]

    if debug_dict is not None:
        debug_dict['weights'] = weights_np
        debug_dict['wdiffs'] = wdiffs_np

    def _compute_alignments(x, lat, pred, source=None, noise_eps=noise_eps_all):
        if source is None:
            idx_wo_pc = [i for i in range(nb_classes) if i != pred]
            assert len(idx_wo_pc) == nb_classes - 1
        else:
            idx_wo_pc = source

        alignments = OrderedDict()
        for neps in noise_eps:
            alignments[neps] = _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps)
            # if debug_dict is not None:
                # debug_dict.setdefault('lat', []).append(lat)
                # debug_dict.setdefault('lat_noisy', []).append(lat_noisy)
                # debug_dict['weights'] = weights
                # debug_dict['wdiffs'] = wdiffs
        return alignments, idx_wo_pc

    def _collect_wdiff_stats(x_set, latent_set, x_preds_set, clean, save_alignments_dir=None, load_alignments_dir=None):
        if clean:
            wdiff_stats = {(tc, tc, e): [] for tc in range(nb_classes) for e in noise_eps_all}
            name = 'clean'
        else:
            wdiff_stats = {(sc, tc, e): [] for sc in range(nb_classes) for tc in range(nb_classes) for e in noise_eps_all if sc != tc}
            name = 'adv'

        def _compute_stats_from_values(v, raw=False):
            if not v.shape:
                return None
            v = v.mean(1)
            if debug:
                v = np.concatenate([v, v*.5, v*1.5])
            if clean or not fit_classifier:
                if v.shape[0] < 3:
                    return None
                return v.mean(0), v.std(0)
            else:
                return v

        for neps in noise_eps_all:
            neps_keys = {k for k in wdiff_stats.keys() if k[-1] == neps}
            loading = load_alignments_dir
            if loading:
                for k in neps_keys:
                    fn = 'alignments_{}_{}.npy'.format(name, str(k))
                    load_fn = os.path.join(load_alignments_dir, fn)
                    if not os.path.exists(load_fn):
                        loading = False
                        break
                    v = np.load(load_fn)
                    wdiff_stats[k] = _compute_stats_from_values(v)
                logging.info('loading alignments from {} for {}'.format(load_alignments_dir, neps))
            if not loading:
                for x, lc, pc, pcc in tqdm.tqdm(zip(x_set, latent_set, x_preds_set, x_preds_clean), total=len(x_set), desc='collecting stats for {}'.format(neps)):
                    if len(lc.shape) == 2:
                        alignments = []
                        for i, (xi, lci, pci) in enumerate(zip(x, lc, pc)):
                            if i == pcc:
                                continue
                            alignments_i, _ = _compute_alignments(xi, lci, i, source=pcc, noise_eps=[neps])
                            for e, a in alignments_i.items():
                                wdiff_stats[(pcc, i, e)].append(a)
                    else:
                        alignments, idx_wo_pc = _compute_alignments(x, lc, pc, noise_eps=[neps])
                        for e, a in alignments.items():
                            wdiff_stats[(pcc, pc, e)].append(a)

                saving = save_alignments_dir and not loading
                if saving:
                    logging.info('saving alignments to {} for {}'.format(save_alignments_dir, neps))
                if debug:
                    some_v = None
                    for k in neps_keys:
                        some_v = some_v or wdiff_stats[k]
                    for k in neps_keys:
                        wdiff_stats[k] = wdiff_stats[k] or some_v

                for k in neps_keys:
                    wdsk = wdiff_stats[k]
                    if len(wdsk):
                        wdiff_stats[k] = np.stack(wdsk)
                    else:
                        wdiff_stats[k] = np.array(None)
                    if saving:
                        fn = 'alignments_{}_{}.npy'.format(name, str(k))
                        save_fn = os.path.join(save_alignments_dir, fn)
                        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                        wds = wdiff_stats[k]
                        np.save(save_fn, wds)
                    wdiff_stats[k] = _compute_stats_from_values(wdiff_stats[k])
        return wdiff_stats

    save_alignments_dir_clean = os.path.join(save_alignments_dir, 'clean') if save_alignments_dir else None
    save_alignments_dir_pgd = os.path.join(save_alignments_dir, 'pgd') if save_alignments_dir else None
    load_alignments_dir_clean = os.path.join(load_alignments_dir, 'clean') if load_alignments_dir else None
    load_alignments_dir_pgd = os.path.join(load_alignments_dir, 'pgd') if load_alignments_dir else None
    if load_alignments_dir:
        load_alignments_dir_clean, load_alignments_dir_pgd = map(lambda s: '{}_{}'.format(s, 'clip' if clip_alignments else 'noclip'), (load_alignments_dir_clean, load_alignments_dir_pgd))
    if save_alignments_dir:
        save_alignments_dir_clean, save_alignments_dir_pgd = map(lambda s: '{}_{}'.format(s, 'clip' if clip_alignments else 'noclip'), (save_alignments_dir_clean, save_alignments_dir_pgd))
    wdiff_stats_clean = _collect_wdiff_stats(x_train, latent_clean, x_preds_clean, clean=True, save_alignments_dir=save_alignments_dir_clean, load_alignments_dir=load_alignments_dir_clean)
    if not just_detect:
        wdiff_stats_pgd = _collect_wdiff_stats(x_train_pgd, latent_pgd, x_preds_pgd, clean=False, save_alignments_dir=save_alignments_dir_pgd, load_alignments_dir=load_alignments_dir_pgd)

    if debug_dict is not None and False:
        esizes = OrderedDict((k, []) for k in noise_eps_all)
        for k, (mc, sc) in wdiff_stats_clean.items():
            mp, sp = wdiff_stats_pgd[k]
            esizes[k[-1]].append(np.abs(mp - mc) / ((sp + sc) / 2.))
        debug_dict['effect_sizes'] = OrderedDict((k, np.array(v)) for k, v in esizes.items())

    wdiff_stats_clean_detect = [np.stack([wdiff_stats_clean[(p, p, eps)] for eps in noise_eps_detect]) for p in range(nb_classes)]
    wdiff_stats_clean_detect = [s.transpose((1, 0, 2)) if len(s.shape) == 3 else None for s in wdiff_stats_clean_detect]
    wdiff_stats_pgd_classify = []
    if not just_detect:
        for tc in range(nb_classes):
            tc_stats = []
            for sc in range(nb_classes):
                if sc == tc:
                    continue
                sc_stats = [wdiff_stats_pgd[(sc, tc, eps)] for eps in noise_eps]
                if sc_stats[0] is None:
                    tc_stats.append(None)
                else:
                    tc_stats.append(np.stack(sc_stats, 1))
            wdiff_stats_pgd_classify.append(tc_stats)

        if fit_classifier:
            logging.info('fitting classifier')
            for tc in tqdm.trange(nb_classes):
                tc_X = []
                tc_Y = []
                idx_wo_tc = [sc for sc in range(nb_classes) if sc != tc]
                for i, sc in enumerate(idx_wo_tc):
                    sc_data = wdiff_stats_pgd_classify[tc][i]
                    if sc_data is not None:
                        sc_data = sc_data.reshape(sc_data.shape[0], -1)
                        for d in sc_data:
                            tc_X.append(d.ravel())
                            tc_Y.append(sc)
                Y_unq = np.unique(tc_Y)
                if len(Y_unq) == 0:
                    lr = SimpleNamespace(predict=lambda x: np.array(tc))
                elif len(Y_unq) == 1:
                    lr = SimpleNamespace(predict=lambda x: np.array(tc_Y[0]))
                else:
                    tc_X = np.stack(tc_X)
                    tc_Y = np.array(tc_Y)
                    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
                    lr.fit(tc_X, tc_Y)
                wdiff_stats_pgd_classify[tc] = lr

    batch = yield

    while batch is not None:
        batch_latent, batch_pred = get_latent_and_pred(batch)
        if debug_dict is not None:
            debug_dict.setdefault('batch_pred', []).append(batch_pred)
        corrected_pred = []
        detection = []
        for b, lb, pb in zip(batch, batch_latent, batch_pred):
            b_align, idx_wo_pb = _compute_alignments(b, lb, pb)
            b_align_det = np.stack([b_align[eps] for eps in noise_eps_detect])
            b_align = np.stack([b_align[eps] for eps in noise_eps])

            wdsc_det_pb = wdiff_stats_clean_detect[pb]
            if wdsc_det_pb is None:
                z_hit = False
            else:
                wdm_det, wds_det = wdsc_det_pb
                z_clean = (b_align_det - wdm_det[:, None]) / wds_det[:, None]
                z_clean_mean = z_clean.mean(1)
                z_cutoff = scipy.stats.norm.ppf(p_ratio_cutoff)
                z_hit = z_clean_mean.mean(0).max(-1) > z_cutoff

            if not just_detect:
                if fit_classifier:
                    lr = wdiff_stats_pgd_classify[pb]
                    b_align = b_align.mean(1).reshape((1, -1))
                    lr_pred = lr.predict(b_align)
                else:
                    wdp = wdiff_stats_pgd_classify[pb]
                    if wdp is None:
                        z_pgd_mode = None
                    else:
                        wdp_not_none_idcs = [i for i, w in enumerate(wdp) if w is not None]
                        if len(wdp_not_none_idcs) == 0:
                            z_pgd_mode = None
                        else:
                            wdp = np.stack([wdp[i] for i in wdp_not_none_idcs], 2)
                            idx_wo_pb_wdp = [idx_wo_pb[i] for i in wdp_not_none_idcs]
                            ssidx = np.arange(wdp.shape[-2])
                            wdp = wdp[:, :, ssidx, ssidx]
                            wdmp, wdsp = wdp
                            b_align = b_align[:, :, wdp_not_none_idcs]
                            z_pgd = (b_align - wdmp[:, None]) / wdsp[:, None]
                            z_pgd_mean = z_pgd.mean(1)
                            z_pgd_mode = scipy.stats.mode(z_pgd_mean.argmax(-1)).mode[0]
            if z_hit:
                if not just_detect:
                    if fit_classifier:
                        print(lr_pred)
                        pb = lr_pred.item()
                    else:
                        if z_pgd_mode is not None:
                            pb = idx_wo_pb_wdp[z_pgd_mode]
                detection.append(True)
            else:
                detection.append(False)
            if debug_dict is not None:
                debug_dict.setdefault('b_align', []).append(b_align)
                # debug_dict.setdefault('stats', []).append((wdm_det, wds_det, wdmp, wdsp))
                # debug_dict.setdefault('p_ratio', []).append(p_ratio)
                # debug_dict.setdefault('p_clean', []).append(p_clean)
                # debug_dict.setdefault('p_pgd', []).append(p_pgd)
                debug_dict.setdefault('z_clean', []).append(z_clean)
                # debug_dict.setdefault('z_conf', []).append(z_conf)
                # debug_dict.setdefault('z_pgdm', []).append(z_pgdm)
                # debug_dict.setdefault('z_pgd', []).append(z_pgd)
            corrected_pred.append(pb)
        if debug_dict is not None:
            debug_dict.setdefault('detection', []).append(detection)
            debug_dict.setdefault('corrected_pred', []).append(corrected_pred)
        batch = yield np.stack((corrected_pred, detection), -1)
