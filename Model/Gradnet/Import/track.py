"""
@InProceedings{GradNet_ICCV2019,
author = {Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, Huchuan Lu},
title = {GradNet: Gradient-Guided Network for Visual Object Tracking},
booktitle = {ICCV},
month = {October},
year = {2019}
}
"""
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from Model.Gradnet.Import.siamese import SiameseNet


def getOpts(opts):
    opts['numScale'] = 3
    opts['scaleStep'] = 1.04
    opts['scalePenalty'] = 0.97
    opts['lossRPos'] = 16
    opts['lossRNeg'] = 0
    opts['scaleLr'] = 0.59
    opts['responseUp'] = 16
    opts['windowing'] = 'cosine'
    opts['wInfluence'] = 0.25
    opts['wInfluence_nosia'] = 0.15
    opts['exemplarSize'] = 127
    opts['instanceSize'] = 255
    opts['scoreSize'] = 17
    opts['totalStride'] = 8
    opts['contextAmount'] = 0.5
    opts['trainWeightDecay'] = 5e-04
    opts['stddev'] = 0.01
    opts['subMean'] = False
    opts['model_path'] = './ckpt/base_l5_1t_49/model_epoch49.ckpt'
    return opts


def getAxisAlignedBB(region):
    region = np.array(region)
    nv = region.size
    assert (nv == 8 or nv == 4)

    if nv == 8:
        xs = region[0:: 2]
        ys = region[1:: 2]
        cx = np.mean(xs)
        cy = np.mean(ys)
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)
        A1 = np.linalg.norm(np.array(region[0:2]) - np.array(region[2:4])) * np.linalg.norm(
            np.array(region[2:4]) - np.array(region[4:6]))
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2

    return cx - 1, cy - 1, w, h


def frameGenerator(vpath):
    imgs = []
    included_extenstions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
    imgFiles = [fn for fn in os.listdir(vpath)
                if any(fn.endswith(ext) for ext in included_extenstions)]
    imgFiles.sort()

    for imgFile in imgFiles:
        img_path = os.path.join(vpath, imgFile)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        imgs.append(img)

    return imgs


def region_to_bbox(region, center=True):
    n = len(region)
    if n == 4:
        return _rect(region, center)


# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return np.array([cx, cy, w, h])
    else:
        # region[0] -= 1
        # region[1] -= 1
        return region


def loadVideoInfo(basePath, video):
    videoPath = os.path.join(basePath, video, 'img')
    if video == 'Human4' or video == 'Human4-2':
        groundTruthFile = os.path.join(basePath, video, 'groundtruth_rect.2.txt')
    elif video == 'Jogging-1' or video == 'Skating2-1':
        groundTruthFile = os.path.join(basePath, video, 'groundtruth_rect.1.txt')
    elif video == 'Jogging-2' or video == 'Skating2-2':
        groundTruthFile = os.path.join(basePath, video, 'groundtruth_rect.2.txt')
    else:
        groundTruthFile = os.path.join(basePath, video, 'groundtruth_rect.txt')
    # groundTruthFile = os.path.join(basePath, video, video + '_gt.txt')
    with open(groundTruthFile) as f:
        gt = np.loadtxt(x.replace(',', ' ') for x in f)

    groundTruth = open(groundTruthFile, 'r')
    reader = groundTruth.readline()
    cx, cy, w, h = region_to_bbox(gt[0])
    # cx, cy, w, h = getAxisAlignedBB(region)
    pos = [cy, cx]
    targetSz = [h, w]

    imgs = frameGenerator(videoPath)
    if video == 'David':
        imgs = imgs[299:]
    # elif video=='Tiger2':
    #     imgs = imgs[6:]
    #     gt = gt[6:]
    elif not imgs.__len__() == gt.shape[0]:
        a = gt.shape[0]
        imgs = imgs[:a]
    # pdb.set_trace()
    assert imgs.__len__() == gt.shape[0]
    return imgs, np.array(pos), np.array(targetSz), gt


def createLogLossLabel(labelSize, rPos, rNeg):
    labelSide = labelSize[0]

    logLossLabel = np.zeros(labelSize, dtype=np.float32, )
    labelOrigin = np.array([np.floor(labelSide / 2), np.floor(labelSide / 2)])

    for i in range(0, labelSide):
        for j in range(0, labelSide):
            distFromOrigin = np.linalg.norm(np.array([i, j]) - labelOrigin)
            if distFromOrigin <= rPos:
                logLossLabel[i, j] = 1
            else:
                if distFromOrigin <= rNeg:
                    logLossLabel[i, j] = 0
                else:
                    logLossLabel[i, j] = -1

    return logLossLabel


def createLabels(labelSize, rPos, rNeg, batchSize):
    half = np.floor(labelSize[0] / 2)

    fixedLabel = createLogLossLabel(labelSize, rPos, rNeg)
    instanceWeight = np.ones(fixedLabel.shape)
    idxP = np.where(fixedLabel == 1)
    idxN = np.where(fixedLabel == -1)

    sumP = len(idxP[0])
    sumN = len(idxN[0])

    # instanceWeight = instanceWeight/225.
    instanceWeight[idxP[0], idxP[1]] = 0.5 * instanceWeight[idxP[0], idxP[1]] / sumP
    instanceWeight[idxN[0], idxN[1]] = 0.5 * instanceWeight[idxN[0], idxN[1]] / sumN

    fixedLabels = np.zeros([batchSize, labelSize[0], labelSize[1], 1], dtype=np.float32)
    instanceWeights = np.zeros([batchSize, labelSize[0], labelSize[1], 1], dtype=np.float32)

    for i in range(batchSize):
        fixedLabels[i, :, :, 0] = fixedLabel
        instanceWeights[i, :, :, 0] = instanceWeight

    return fixedLabels, instanceWeights


def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):
    if originalSz is None:
        originalSz = modelSz

    sz = originalSz
    im_sz = img.shape
    # make sure the size is not too small
    assert min(im_sz[:2]) > 2, "the size is too small"
    c = (np.array(sz) + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = round(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1] + 1))
    bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))

    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)

    if top_pad or left_pad or bottom_pad or right_pad:
        r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[0])
        g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[1])
        b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[2])
        r = np.expand_dims(r, 2)
        g = np.expand_dims(g, 2)
        b = np.expand_dims(b, 2)

        # h, w = r.shape
        # r1 = np.zeros([h, w, 1], dtype=np.float32)
        # r1[:, :, 0] = r
        # g1 = np.zeros([h, w, 1], dtype=np.float32)
        # g1[:, :, 0] = g
        # b1 = np.zeros([h, w, 1], dtype=np.float32)
        # b1[:, :, 0] = b

        img = np.concatenate((r, g, b), axis=2)

    im_patch_original = img[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
    if not np.array_equal(modelSz, originalSz):
        im_patch = cv2.resize(im_patch_original, modelSz)
        # im_patch_original = im_patch_original/255.0
        # im_patch = transform.resize(im_patch_original, modelSz)*255.0
        # im = Image.fromarray(im_patch_original.astype(np.float))
        # im = im.resize(modelSz)
        # im_patch = np.array(im).astype(np.float32)
    else:
        im_patch = im_patch_original

    return im_patch, im_patch_original


def makeScalePyramid(im, targetPosition, in_side_scaled, out_side, avgChans, stats, p):
    """
    computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
    and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.

    """
    in_side_scaled = np.round(in_side_scaled)
    max_target_side = int(round(in_side_scaled[-1]))
    min_target_side = int(round(in_side_scaled[0]))
    beta = out_side / float(min_target_side)
    # size_in_search_area = beta * size_in_image
    # e.g. out_side = beta * min_target_side
    search_side = int(round(beta * max_target_side))
    search_region, _ = getSubWinTracking(im, targetPosition, (search_side, search_side),
                                         (max_target_side, max_target_side), avgChans)
    if p['subMean']:
        pass
    assert round(beta * min_target_side) == int(out_side)

    tmp_list = []
    tmp_pos = ((search_side - 1) / 2., (search_side - 1) / 2.)
    for s in range(p['numScale']):
        target_side = round(beta * in_side_scaled[s])
        tmp_region, _ = getSubWinTracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side),
                                          avgChans)
        tmp_list.append(tmp_region)

    pyramid = np.stack(tmp_list)

    return pyramid


def trackerEval(score, score_nosia, sx, targetPosition, window, opts):
    # responseMaps = np.transpose(score[:, :, :, 0], [1, 2, 0])
    responseMaps = score[:, :, :, 0]
    responseMaps_nosia = score_nosia[:, :, :, 0]
    upsz = opts['scoreSize'] * opts['responseUp']
    # responseMapsUp = np.zeros([opts['scoreSize']*opts['responseUp'], opts['scoreSize']*opts['responseUp'], opts['numScale']])
    responseMapsUP = []

    if opts['numScale'] > 1:
        currentScaleID = int(opts['numScale'] / 2)
        bestScale = currentScaleID
        bestPeak = -float('Inf')

        for s in range(opts['numScale']):
            if opts['responseUp'] > 1:
                responseMapsUP.append(cv2.resize(responseMaps[s, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC))
            else:
                responseMapsUP.append(responseMaps[s, :, :])

            thisResponse = responseMapsUP[-1]

            if s != currentScaleID:
                thisResponse = thisResponse * opts['scalePenalty']

            thisPeak = np.max(thisResponse)
            if thisPeak > bestPeak:
                bestPeak = thisPeak
                bestScale = s

        responseMap = responseMapsUP[bestScale]
    else:
        responseMap = cv2.resize(responseMaps[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
        bestScale = 0

    responseMaps_nosia = cv2.resize(responseMaps_nosia[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
    responseMaps_nosia = responseMaps_nosia - np.min(responseMaps_nosia)
    responseMaps_nosia = responseMaps_nosia / np.sum(responseMaps_nosia)

    responseMap = responseMap - np.min(responseMap)
    responseMap = responseMap / np.sum(responseMap)

    responseMap = (1 - opts['wInfluence_nosia']) * responseMap + opts['wInfluence_nosia'] * responseMaps_nosia

    responseMap = (1 - opts['wInfluence']) * responseMap + opts['wInfluence'] * window

    # responseMap = (1 - opts['wInfluence']) * responseMap + opts['wInfluence'] * window
    # responseMap = (1 - opts['wInfluence_nosia']) * responseMap + opts['wInfluence_nosia'] * responseMaps_nosia

    rMax, cMax = np.unravel_index(responseMap.argmax(), responseMap.shape)
    pCorr = np.array((rMax, cMax))
    dispInstanceFinal = pCorr - int(upsz / 2)
    dispInstanceInput = dispInstanceFinal * opts['totalStride'] / opts['responseUp']
    dispInstanceFrame = dispInstanceInput * sx / opts['instanceSize']
    newTargetPosition = targetPosition + dispInstanceFrame
    # print(bestScale)

    return newTargetPosition, bestScale


def get_sequence(data_dir, seq_name):
    # generate config from a sequence name
    img_dir = os.path.join(data_dir, seq_name, 'img')
    gt_path = os.path.join(data_dir, seq_name, 'groundtruth_rect.txt')
    included_extenstions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
    img_list = [fn for fn in os.listdir(img_dir)
                if any(fn.endswith(ext) for ext in included_extenstions)]
    # file_names.sort()
    # img_num = len(file_names)
    # img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir, x) for x in img_list]
    with open(gt_path) as f:
        gt = np.loadtxt((x.replace(',', ' ') for x in f))

    n_frames = len(img_list)
    if not n_frames == len(gt):
        img_list = img_list[0:len(gt)]
        n_frames = len(gt)
    # gt = np.loadtxt(gt_path,delimiter=',')
    init_bbox = gt[0]
    return img_list, init_bbox, gt


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = np.float(sum(new_distances < dist_threshold)) / np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds + 1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = np.float(sum(new_distances < thresholds[i])) / np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2))
    b = np.array((boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return np.float(iou)


'''----------------------------------------main-----------------------------------------------------'''


def run_siamesefc(opts, exemplarOp_init, instanceOp_init, instanceOp, zFeat2Op_gra, zFeat5Op_gra, zFeat5Op_sia,
                  scoreOp_sia, scoreOp_gra, zFeat2Op_init, sess, display):
    # nImgs = len(imgs)
    # startFrame = 0
    # im = imgs[startFrame]
    my_img, tracking_object = yield
    targetPosition, targetSize = tracking_object
    im = my_img

    avgChans = np.mean(im, axis=(
        0, 1))  # [np.mean(np.mean(img[:, :, 0])), np.mean(np.mean(img[:, :, 1])), np.mean(np.mean(img[:, :, 2]))]
    wcz = targetSize[1] + opts['contextAmount'] * np.sum(targetSize)
    hcz = targetSize[0] + opts['contextAmount'] * np.sum(targetSize)
    sz = np.sqrt(wcz * hcz)
    scalez = opts['exemplarSize'] / sz

    zCrop, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']),
                                 (np.around(sz), np.around(sz)), avgChans)

    if opts['subMean']:
        pass

    dSearch = (opts['instanceSize'] - opts['exemplarSize']) / 2
    pad = dSearch / scalez
    sx = sz + 2 * pad

    minSx = 0.2 * sx
    maxSx = 5.0 * sx

    winSz = opts['scoreSize'] * opts['responseUp']
    if opts['windowing'] == 'cosine':
        hann = np.hanning(winSz).reshape(winSz, 1)
        window = hann.dot(hann.T)
    elif opts['windowing'] == 'uniform':
        window = np.ones((winSz, winSz), dtype=np.float32)

    window = window / np.sum(window)
    scales = np.array([opts['scaleStep'] ** i for i in range(int(np.ceil(opts['numScale'] / 2.0) - opts['numScale']),
                                                             int(np.floor(opts['numScale'] / 2.0) + 1))])

    '''initialization at the first frame'''
    xCrops = makeScalePyramid(im, targetPosition, sx * scales, opts['instanceSize'], avgChans, None, opts)
    xCrops0 = np.expand_dims(xCrops[1], 0)
    zCrop = np.expand_dims(zCrop, axis=0)
    zCrop0 = np.copy(zCrop)

    zFeat5_gra_init, zFeat2_gra_init, zFeat5_sia_init = sess.run([zFeat5Op_gra, zFeat2Op_gra, zFeat5Op_sia],
                                                                 feed_dict={exemplarOp_init: zCrop0,
                                                                            instanceOp_init: xCrops0,
                                                                            instanceOp: xCrops})
    template_gra = np.copy(zFeat5_gra_init)
    template_sia = np.copy(zFeat5_sia_init)
    hid_gra = np.copy(zFeat2_gra_init)

    tic = time.time()
    train_all = []
    frame_all = []
    F_max_all = 0
    A_all = []
    F_max_thred = 0
    F_max = 0
    train_all.append(xCrops0)
    A_all.append(0)
    frame_all.append(0)
    updata_features = []
    updata_features_score = []
    updata_features_frame = []
    no_cos = 1
    refind = 0

    i = 0
    my_img = yield
    im = my_img
    while True:
        if i != 0:
            if i - updata_features_frame[-1] == 9 and no_cos:
                opts['wInfluence'] = 0
                no_cos = 0
            else:
                opts['wInfluence'] = 0.25

            if (im.shape[-1] == 1):
                tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
                tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
                im = tmp

            scaledInstance = sx * scales
            scaledTarget = np.array([targetSize * scale for scale in scales])

            xCrops = makeScalePyramid(im, targetPosition, scaledInstance, opts['instanceSize'], avgChans, None, opts)

            score_gra, score_sia = sess.run([scoreOp_gra, scoreOp_sia],
                                            feed_dict={zFeat5Op_gra: template_gra,
                                                       zFeat5Op_sia: template_sia,
                                                       instanceOp: xCrops})
            # sio.savemat('score.mat', {'score': score})
            # score_gra = np.copy(np.expand_dims(score_sia[1],0))

            newTargetPosition, newScale = trackerEval(score_sia, score_gra, round(sx), targetPosition, window, opts)

            targetPosition = newTargetPosition
            sx = max(minSx, min(maxSx, (1 - opts['scaleLr']) * sx + opts['scaleLr'] * scaledInstance[newScale]))
            F_max = np.max(score_sia)
            targetSize = (1 - opts['scaleLr']) * targetSize + opts['scaleLr'] * scaledTarget[newScale]
            # print('frame:%d--loss:%f--frame_now:%d' %(i, np.max(score),frame_now))

            if refind:

                xCrops = makeScalePyramid(im, np.array([im.shape[0] / 2, im.shape[1] / 2]), scaledInstance,
                                          opts['instanceSize'], avgChans, None,
                                          opts)

                score_gra, score_sia = sess.run([scoreOp_gra, scoreOp_sia],
                                                feed_dict={zFeat5Op_gra: template_gra,
                                                           zFeat5Op_sia: template_sia,
                                                           instanceOp: xCrops})
                F_max2 = np.max(score_sia)
                F_max3 = np.max(score_gra)
                if F_max2 > F_max and F_max3 > F_max:
                    newTargetPosition, newScale = trackerEval(score_sia, score_gra, round(sx),
                                                              np.array([im.shape[0] / 2, im.shape[1] / 2]), window,
                                                              opts)

                    targetPosition = newTargetPosition
                    sx = max(minSx, min(maxSx, (1 - opts['scaleLr']) * sx + opts['scaleLr'] * scaledInstance[newScale]))
                    F_max = np.max(score_sia)
                    targetSize = (1 - opts['scaleLr']) * targetSize + opts['scaleLr'] * scaledTarget[newScale]

                refind = 0

            '''use the average of the first five frames to set the threshold'''
            if i < 6:
                F_max_all = F_max_all + F_max
            if i == 5:
                F_max_thred = F_max_all / 5.
        else:
            pass

        '''tracking results'''
        rectPosition = targetPosition - targetSize / 2.
        Position_now = np.concatenate(
            [np.round(rectPosition).astype(int)[::-1], np.round(targetSize).astype(int)[::-1]], 0)

        if Position_now[0] + Position_now[2] > im.shape[1] and F_max < F_max_thred * 0.5:
            refind = 1

        '''if you want use groundtruth'''

        # region = np.copy(gt[i])

        # cx, cy, w, h = getAxisAlignedBB(region)
        # pos = np.array([cy, cx])
        # targetSz = np.array([h, w])
        # iou_ = _compute_distance(region, Position_now)
        #

        '''save the reliable training sample'''
        if F_max >= min(F_max_thred * 0.5, np.mean(updata_features_score)):
            scaledInstance = sx * scales
            xCrops = makeScalePyramid(im, targetPosition, scaledInstance, opts['instanceSize'], avgChans, None, opts)
            updata_features.append(xCrops)
            updata_features_score.append(F_max)
            updata_features_frame.append(i)
            if updata_features_score.__len__() > 5:
                del updata_features_score[0]
                del updata_features[0]
                del updata_features_frame[0]
        else:
            if i < 10 and F_max < F_max_thred * 0.4:
                scaledInstance = sx * scales
                xCrops = makeScalePyramid(im, targetPosition, scaledInstance, opts['instanceSize'], avgChans, None,
                                          opts)
                template_gra, zFeat2_gra = sess.run([zFeat5Op_gra, zFeat2Op_gra],
                                                    feed_dict={zFeat2Op_init: hid_gra,
                                                               instanceOp_init: np.expand_dims(xCrops[1], 0)})
                hid_gra = np.copy(0.3 * hid_gra + 0.7 * zFeat2_gra)

        '''update the template every 5 frames'''

        if i % 5 == 0:
            template_gra, zFeat2_gra = sess.run([zFeat5Op_gra, zFeat2Op_gra],
                                                feed_dict={zFeat2Op_init: hid_gra,
                                                           instanceOp_init: np.expand_dims(
                                                               updata_features[np.argmax(updata_features_score)][1],
                                                               0)})
            hid_gra = np.copy(0.4 * hid_gra + 0.6 * zFeat2_gra)

        my_img = yield np.copy(Position_now)
        im = my_img
        i += 1


def just_show():
    # 设置输出信息的屏蔽级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['C_CPP_MIN_LOG_LEVEL'] = '3'
    # opts:设置字典
    opts = getOpts(dict())

    '''define input tensors and network'''
    exemplarOp_init = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
    instanceOp_init = tf.placeholder(tf.float32, [1, opts['instanceSize'], opts['instanceSize'], 3])
    instanceOp = tf.placeholder(tf.float32, [3, opts['instanceSize'], opts['instanceSize'], 3])
    template_Op = tf.placeholder(tf.float32, [1, 6, 6, 256])
    search_tr_Op = tf.placeholder(tf.float32, [3, 22, 22, 32])
    isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
    lr = tf.constant(0.0001, dtype='float32')
    sn = SiameseNet()

    '''build the model'''
    # initial embedding
    with tf.variable_scope('siamese') as scope:
        zFeat2Op_init, zFeat5Op_init = sn.extract_gra_fea_template(exemplarOp_init, opts, isTrainingOp)
        scoreOp_init = sn.response_map_cal(instanceOp_init, zFeat5Op_init, opts, isTrainingOp)
    # gradient calculation
    labels = np.ones([8], dtype=np.float32)
    respSz = int(scoreOp_init.get_shape()[1])
    respSz = [respSz, respSz]
    respStride = 8
    fixedLabel, instanceWeight = createLabels(respSz, opts['lossRPos'] / respStride, opts['lossRNeg'] / respStride, 1)
    instanceWeightOp = tf.constant(instanceWeight, dtype=tf.float32)
    yOp = tf.constant(fixedLabel, dtype=tf.float32)
    with tf.name_scope("logistic_loss"):
        lossOp_init = sn.loss(scoreOp_init, yOp, instanceWeightOp)
    grad_init = tf.gradients(lossOp_init, zFeat2Op_init)
    # template update and get score map
    with tf.variable_scope('siamese') as scope:
        zFeat5Op_gra, zFeat2Op_gra = sn.template_update_based_grad(zFeat2Op_init, grad_init[0], opts, isTrainingOp)
        scope.reuse_variables()
        zFeat5Op_sia = sn.extract_sia_fea_template(exemplarOp_init, opts, isTrainingOp)
        scoreOp_sia = sn.response_map_cal(instanceOp, zFeat5Op_sia, opts, isTrainingOp)
        scoreOp_gra = sn.response_map_cal(tf.expand_dims(instanceOp[1], 0), zFeat5Op_gra, opts, isTrainingOp)

    '''restore pretrained network'''
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # saver.restore(sess, opts['model_path'])
    saver.restore(sess, r'C:\tem\PycharmProjects\ObjectTracking\Model\Gradnet\Import\ckpt\base_l5_1t_49\model_epoch49.ckpt')


    n_seq = 1
    speed = np.zeros(n_seq)
    precisions = np.zeros(n_seq)
    precisions_auc = np.zeros(n_seq)
    ious = np.zeros(n_seq)
    lengths = np.zeros(n_seq)

    '''tracking process'''
    i = 0
    idx = i
    # print(seq)
    tem = run_siamesefc(opts, exemplarOp_init, instanceOp_init, instanceOp,
                        zFeat2Op_gra, zFeat5Op_gra, zFeat5Op_sia, scoreOp_sia,
                        scoreOp_gra, zFeat2Op_init, sess, display=False)
    next(tem)
    return tem


if __name__ == '__main__':
    test = just_show()
    imgs, targetPosition, targetSize, gt = loadVideoInfo("./dataset/", 'Walking2')
    test.send((imgs[0], (targetPosition, targetSize)))
    for i in imgs[1:]:
        print(test.send(i))
