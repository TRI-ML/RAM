from .generic_dataset import GenericDataset
import os
import numpy as np
import random
from utils.image import get_affine_transform, affine_transform
import copy
import torch.nn.functional as F
import torch
import math
from torchvision.ops.boxes import box_iou
from utils.image import gaussian_radius, draw_umich_gaussian

from utils.pose import Pose

class VideoDataset(GenericDataset):
    
    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
        self.occlusion_thresh = opt.occlusion_thresh
        self.visibility_thresh = opt.visibility_thresh
        self.radius_growth_step = opt.radius_growth_step
        self.variable_stride = opt.variable_stride
        self.input_len = None
        self.min_frame_dist = None
        self.max_frame_dist = None
        self.box_size_thresh = None
        self.min_frame_dist = None
        self.max_frame_dist = None
        self.const_v_over_occl = False
        super(VideoDataset, self).__init__(opt, split, ann_path, img_dir)
        if split != 'train':
            self.input_len = 1

    def hide_occlusions(self, anns):
        for i, frame_anns in enumerate(anns):
            for ann in frame_anns:
                if (ann['occlusion'] < self.occlusion_thresh) and (ann['occlusion'] > self.visibility_thresh):
                    ann['iscrowd'] = 1
                elif ann['occlusion'] <= self.visibility_thresh:
                    ann['iscrowd'] = 2

        return anns

    def project_3dbbox(self, center, intrinsics):
        bbox_proj = np.dot(np.array(intrinsics), np.array(center))
        x = bbox_proj[0] / bbox_proj[2]
        y = bbox_proj[1] / bbox_proj[2]
        return 0.5 * x, np.max(0.5 * y - 158, 0)

    def process_occlusions(self, anns, img_infos, trans_out, h, w, flipped=False):
        track_vis = {}
        tracks = {}
        vis_anns = []
        invis_anns = []
        visibility_range = 2
        prev_scene_pos = None
        for i, frame_anns in enumerate(anns):
            filtered_frame_anns = []
            img_info = img_infos[i]
            if 'pose_quat' in img_info:
                scene_pose = Pose(np.array(img_info['pose_quat']), np.array(img_info['pose_tvec']))

            for ann in frame_anns:
                if ann['track_id'] not in tracks:
                    tracks[ann['track_id']] = [None] * len(anns)
                 
                if 'pose_quat' in ann:
                    ann['pose'] = Pose(np.array(ann['pose_quat']), np.array(ann['pose_tvec']))
                    ann['pose'] = scene_pose * ann['pose']

                tracks[ann['track_id']][i] = ann

                if i > 0 and 'pose' in ann:
                    ann = self.assign_speed(ann, tracks[ann['track_id']][i - 1])
                    ann = self.assign_3d_speed(ann, tracks[ann['track_id']][i - 1])
                else:
                    ann['v'] = np.zeros(2, np.float32)
                    ann['3dv'] = np.zeros(3, np.float32)

                occlusion_thresh = self.occlusion_thresh
                # if ann['category_id'] == 2:
                #     occlusion_thresh *= 2
                track_id = ann['track_id']
                visible = True
                cls_id = int(self.cat_ids[ann['category_id']])
                box = ann['bbox']
                box_size = box[2] * box[3]
                if track_id not in track_vis:
                    track_vis[track_id] = [None] * len(anns)
                if ann['occlusion'] < occlusion_thresh:
                    visible = False

                if 'modal_bbox' in ann:
                    modal_bbox = ann['modal_bbox']
                    if flipped:
                        modal_bbox = [w - modal_bbox[0] - 1 - modal_bbox[2], modal_bbox[1], modal_bbox[2], modal_bbox[3]]
                    bbox, _, truncated = self._get_bbox_output(modal_bbox, trans_out, h, w)
                    box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if box_h * box_w < 10:
                        visible = False

                track_vis[track_id][i] = visible

                if (i < visibility_range) and visible:
                    filtered_frame_anns.append(ann)
                elif (i < visibility_range) and not visible and (ann['occlusion'] > self.visibility_thresh):
                    ann['iscrowd'] = 1
                    filtered_frame_anns.append(ann)
                elif (i < visibility_range) and (ann['occlusion'] <= self.visibility_thresh):
                    ann['iscrowd'] = 2
                
            if i < visibility_range:
                vis_anns.append(filtered_frame_anns)
                invis_anns.append([])

        # number of frames for which the object has been occluded
        invis_count = {}
        for i in range(visibility_range, len(anns)):
            frame_anns = anns[i]
            vis_frame_anns = []
            invis_frame_anns = []
            for ann in frame_anns:
                track_id = ann['track_id']

                if track_vis[track_id][i]:
                    invis_count[track_id] = 0
                    vis_frame_anns.append(ann)
                    continue

                previously_seen = track_vis[track_id][i - 2] and track_vis[track_id][i - 1]
                if previously_seen:
                    if not track_vis[track_id][i]:
                        track_vis[track_id][i] = True

                        if track_id not in invis_count:
                            invis_count[track_id] = 0
                        invis_count[track_id] += 1
                        ann['radius_mutiplier'] = 1 + invis_count[track_id] * self.radius_growth_step
                        invis_frame_anns.append(ann)
                    else:
                        invis_count[track_id] = 0
                        vis_frame_anns.append(ann)
                elif not track_vis[track_id][i] and (ann['occlusion'] > self.visibility_thresh):
                    ann['iscrowd'] = 1
                    vis_frame_anns.append(ann)

            vis_anns.append(vis_frame_anns)
            invis_anns.append(invis_frame_anns)
        
        return vis_anns, invis_anns

    def get_ann_by_id(self, anns, track_id):
        for ann in anns:
            if ann['track_id'] == track_id:
                return ann
                
        return None

    def assign_speed(self, ann, prev_ann):
        if prev_ann is None:
            v = np.zeros(2, np.float32)
        else:
            bbox = self._coco_box_to_bbox(ann['bbox']).copy()
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            bbox = self._coco_box_to_bbox(prev_ann['bbox']).copy()
            prev_ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            v = ct - prev_ct

        ann['v'] = v

        return ann

    def assign_3d_speed(self, ann, prev_ann):
        if prev_ann is None:
            v = np.zeros(3, np.float32)
        else:
            v = ann['pose'].tvec - prev_ann['pose'].tvec

        ann['3dv'] = v

        return ann

    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    # for an invisible object generates a heatmap with centers of visible objects to supervise the overlap avoidance objective 
    def init_occl_mask(self, mask_indices, mask_weights, id_to_center, self_id, k, rw_down_rate):
        rw_label = np.zeros((self.default_resolution[0] // (self.opt.down_ratio * rw_down_rate), self.default_resolution[1] // (self.opt.down_ratio * rw_down_rate)), np.float32)
        for track_id, ct_int in id_to_center.items():
            # do not include our own center
            if track_id == self_id or track_id == -1:
                continue

            draw_umich_gaussian(rw_label, ct_int, 1)

        rw_label = np.reshape(rw_label, (rw_label.shape[0] * rw_label.shape[1]))
        indices = rw_label.nonzero()[0]
        weights = np.zeros((1))
        if len(indices) != 0:
            weights = rw_label[indices]

        indices = indices[:mask_indices.shape[1]]
        mask_indices[k][:indices.shape[0]] = indices

        weights = weights[:mask_weights.shape[1]]
        mask_weights[k][:weights.shape[0]] = weights

        return mask_indices, mask_weights


    # Generete groud truth node labels for visible objects as well as some additional info
    def get_rw_labels(self, id_to_center, id_to_size, id_to_slot, id_to_disp, id_to_idx, rw_inds, rw_labels, rw_mask, mask_indices, mask_weights, rw_down_rate, id_to_occllen, seen_ids, frame_ind):
        k = 0
        for track_id, idx in id_to_slot.items():
            k += 1
            rw_inds[k] = idx[0]
            # If an object is invisible in this frame but was visible in the past, we might still want to supervise the Walker to avoid the centers of visible objects
            # Also, if at some point in the video an object experiences a dispalcement that is larger than the radius of the locall attention then we simlpy skip this object for the entire sequence
            if track_id not in id_to_center or (id_to_disp[track_id] >= (self.opt.local_rw_r * self.default_resolution[0] // self.opt.down_ratio)):
                rw_mask[k] = 0
                rw_labels[k] = 0
                if track_id in seen_ids:
                    # Compute the length of the occlusion for invisible instances
                    id_to_occllen[track_id] += 1
                    if self.opt.sup_centeroverlap:
                        mask_indices, mask_weights = self.init_occl_mask(mask_indices, mask_weights, id_to_center, track_id, k, rw_down_rate)
                continue

            # First frame in which the object becomes visible, nothing to supervise yet
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                continue
                
            # The weight of the Walker loss is proportional to the length of the occlusion
            rw_mask[k] = 20 * id_to_occllen[track_id] + 1
            id_to_occllen[track_id] = 0
            
            # Labels are represented as heatmaps. In practice, only the center of the heatmap is used for supervising the Walker. The rest of the heatmap is used to remove hard negatives around the center.
            rw_label = np.zeros((self.default_resolution[0] // (self.opt.down_ratio * rw_down_rate), self.default_resolution[1] // (self.opt.down_ratio * rw_down_rate)), np.float32)
            w, h = id_to_size[track_id]
            radius = gaussian_radius((math.ceil(h / rw_down_rate), math.ceil(w / rw_down_rate)))
            radius = max(0, int(radius))
            ct_int = id_to_center[track_id]
            draw_umich_gaussian(rw_label, ct_int, radius)
            rw_labels[k] = np.reshape(rw_label, (1, rw_label.shape[0] * rw_label.shape[1]))

        return rw_inds, rw_labels, rw_mask, mask_indices, mask_weights, id_to_occllen, seen_ids


    # Converts box centers to node indices for visible objects only, plus extracts some additional infor for future processing
    def get_idx_id_maps(self, frame_batch, gt_det):
        id_to_idx = {}
        id_to_size = {}
        id_to_center = {}
        id_to_disp = {}
        rw_downsample = math.floor(self.opt.pool_kernel / 2) + 1
        for i in range(len(frame_batch['mask'])):
            if frame_batch['mask'][i] == 0 or ('occlusion' in gt_det and gt_det['occlusion'][i] is not None and gt_det['occlusion'][i] <= self.opt.visibility_thresh) or (frame_batch['track_ids'][i] == -1):
                continue

            track_id = frame_batch['track_ids'][i]
            ct = [frame_batch['reg'][i][0] + frame_batch['ct_int'][i][0], frame_batch['reg'][i][1] + frame_batch['ct_int'][i][1]]
            ct_int = [int(x / rw_downsample) for x in ct]

            ind = ct_int[1] * (self.default_resolution[1] // (self.opt.down_ratio * rw_downsample)) + ct_int[0]

            if track_id not in id_to_idx:
                id_to_idx[track_id] = []
            id_to_idx[track_id].append(ind)
            id_to_size[track_id] = frame_batch['wh'][i]
            id_to_center[track_id] = ct_int

            disp = frame_batch['tracking'][i]
            id_to_disp[track_id] = math.sqrt(disp[0] ** 2 + disp[1] ** 2)

        return id_to_idx, id_to_size, id_to_center, id_to_disp

    def check_overlap(self, id_to_center, new_keys):
        centers_tensor = None
        ignored = []
        for track_id, ct_int in id_to_center.items():
            if centers_tensor is not None:
                centers_tensor = torch.cat((centers_tensor, torch.Tensor(ct_int).view(1, 2).float()), dim = 0)
            else:
                centers_tensor = torch.Tensor(ct_int).view(1, 2).float()

        if centers_tensor is None or len(centers_tensor) < 2:
            return ignored

        for new_track_id in new_keys:
            ct_int = id_to_center[new_track_id]
            dsts = torch.cdist(torch.Tensor(ct_int).view(1, 2).float(), centers_tensor)
            v, i = torch.kthvalue(dsts, 2)
            if v <= 1:
                ignored.append(new_track_id)

        return ignored

    def __getitem__(self, index):
        opt = self.opt
        if self.input_len == None:
            self.input_len = opt.input_len
        imgs, anns, img_infos, img_paths = self._load_data(index, self.input_len)

        height, width = imgs[0].shape[0], imgs[0].shape[1]
        c = np.array([imgs[0].shape[1] / 2., imgs[0].shape[0] / 2.], dtype=np.float32)
        s = max(imgs[0].shape[0], imgs[0].shape[1]) * 1.0 if not self.opt.not_max_crop \
        else np.array([imgs[0].shape[1], imgs[0].shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0
        if self.split == 'train':
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
        s = s * aug_s
        if np.random.random() < opt.flip:
            flipped = 1

        trans_input = get_affine_transform(
        c, s, rot, [self.default_resolution[1], self.default_resolution[0]])
        trans_output = get_affine_transform(
        c, s, rot, [self.default_resolution[1] // self.opt.down_ratio, self.default_resolution[0] // self.opt.down_ratio])
 
        invis_anns = [[]] * len(anns)
        if (len(anns[0]) > 0) and 'occlusion' in anns[0][0] and opt.sup_invis:
            anns, invis_anns = self.process_occlusions(copy.deepcopy(anns), img_infos, trans_output, height, width, flipped)
        elif (len(anns[0]) > 0) and 'occlusion' in anns[0][0] and not opt.sup_invis:
            anns = self.hide_occlusions(copy.deepcopy(anns))

        rets = []
        pre_anns = copy.deepcopy(anns[0])
        pre_invis_anns = copy.deepcopy(anns[0])
        apply_noise_to_centers = False
        skipped_objs = {}
        out_of_frame = set([])
        id_to_slot = None
        id_to_occlen = {}
        seen_ids = set([])
        for i in range(len(imgs)):
            # do not disturt pre centers in the first frame to avoid non-zero tracking targets
            if i > 0:
                apply_noise_to_centers = True
            img = imgs[i]
            anns_frame = copy.deepcopy(anns[i])
            anns_invis_frame = copy.deepcopy(invis_anns[i])
            img_info = img_infos[i]
            if flipped:
                img = img[:, ::-1, :]
                anns_frame = self._flip_anns(anns_frame, width)
                anns_invis_frame = self._flip_anns(anns_invis_frame, width)
                pre_anns = self._flip_anns(pre_anns, width)
                pre_invis_anns = self._flip_anns(pre_invis_anns, width)

            pre_invis_anns = self.update_pre(pre_invis_anns, skipped_objs)
            pre_anns.extend(pre_invis_anns)

            pre_hm, pre_cts, track_ids, occl_lengths, pre_vs, pre_ids = self._get_pre_dets(
                pre_anns, trans_input, trans_output, apply_noise_to_centers)

            if i > 0 and not self.same_aug_pre and self.split == 'train':
                c, aug_s, _ = self._get_aug_param(c, s, width, height, disturb=True)
                s = s * aug_s
                trans_input = get_affine_transform(
                  c, s, rot, [self.default_resolution[1], self.default_resolution[0]])
                trans_output = get_affine_transform(
                  c, s, rot, [self.default_resolution[1] // self.opt.down_ratio, self.default_resolution[0] // self.opt.down_ratio])

            inp = self._get_input(img, trans_input, self.mean, self.std)
            ret = {'image': inp}
            gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}
            
            ### init samples
            self._init_ret(ret, gt_det)
            calib = self._get_calib(img_info, width, height)
            
            num_objs = min(len(anns_frame), self.max_objs)
            counter = 0
            for k in range(num_objs):
                ann = anns_frame[k]
                radius_mutiplier = None
                occlusion = None
                if 'occlusion' in ann:
                    occlusion = ann['occlusion']
                cls_id = int(self.cat_ids[ann['category_id']])
                if cls_id > self.opt.num_classes or cls_id <= -999:
                    continue
                bbox, bbox_amodal, truncated = self._get_bbox_output(
                    ann['bbox'], trans_output, height, width)
                orig_box = ann['bbox']
                box_size = orig_box[2] * orig_box[3]
                if ('iscrowd' in ann) and (ann['iscrowd'] == 2):
                    continue

                if ann['track_id'] in out_of_frame:
                    out_of_frame.remove(ann['track_id'])
                elif (cls_id <= 0) or (('iscrowd' in ann) and (ann['iscrowd'] == 1)) or (box_size < self.box_size_thresh[cls_id - 1]):
                    v = self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids, gt_det)
                    continue

                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                is_added = self._add_instance(
                    ret, gt_det, counter, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                    calib, pre_cts, track_ids, pre_vs, pre_ids, occlusion, radius_mutiplier, occl_lengths)
                if is_added:
                    counter += 1

            num_objs = min(len(anns_invis_frame), self.max_objs)
            prev_skipped_objs = skipped_objs
            skipped_objs = {}
            for k in range(num_objs):
                ann = anns_invis_frame[k]

                bbox, bbox_amodal, truncated = self._get_bbox_output(
                        ann['bbox'], trans_output, height, width)
                orig_box = ann['bbox']
                box_size = orig_box[2] * orig_box[3]

                cls_id = int(self.cat_ids[ann['category_id']])
                if cls_id > self.opt.num_classes or cls_id <= 0:
                    continue

                if self.const_v_over_occl:
                    if ann['track_id'] in out_of_frame:
                        continue

                    gt_box = ann['bbox']
                    gt_occl = ann['occlusion']

                    if ann['track_id'] in prev_skipped_objs:
                        ann = prev_skipped_objs[ann['track_id']]
                    else:
                        ann = self.get_ann_by_id(pre_anns, ann['track_id'])
                        ann['last_seen_size'] = ann['bbox'][2] * ann['bbox'][2] 
                        if ann['track_id'] not in track_ids:
                            out_of_frame.add(ann['track_id'])
                            continue

                    last_seen_size = ann['last_seen_size']

                    if 'radius_mutiplier' not in ann:
                        ann['radius_mutiplier'] = 1

                    radius_mutiplier = ann['radius_mutiplier']

                    ann['pose'].tvec += ann['3dv']
                    frame_3dpose = scene_pose.inverse() * ann['pose'] 
                    curr_x, curr_y = self.project_3dbbox(frame_3dpose.tvec, scene_intrinsics)

                    if (curr_x <= 0) or (curr_y <= 0) or (curr_x >= (self.default_resolution[1] - 1)) or (curr_y >= (self.default_resolution[0] - 1)):
                        bbox, bbox_amodal, truncated = self._get_bbox_output(gt_box, trans_output, height, width)
                        if (bbox[2] - bbox[0]) <= 1 or (bbox[3] - bbox[1]) <= 1:
                            out_of_frame.add(ann['track_id'])
                            continue

                        self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids, gt_det)
                        skipped_objs[ann['track_id']] = ann
                        continue

                    new_box = np.array([curr_x - gt_box[2] / 2, curr_y - gt_box[3] / 2, gt_box[2], gt_box[3]])
                    ann['bbox'] = new_box

                    if flipped:
                        bbox = ann['bbox']
                        ann['bbox'] = [
                            width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

                    if ann['bbox'][0] < 0:
                        ann['bbox'][2] += ann['bbox'][0]
                        ann['bbox'][0] = 0

                    if ann['bbox'][1] < 0:
                        ann['bbox'][3] += ann['bbox'][1]
                        ann['bbox'][1] = 0

                    if (ann['bbox'][0] + ann['bbox'][2]) > self.opt.input_w:
                        ann['bbox'][2] += self.opt.input_w - (ann['bbox'][0] + ann['bbox'][2])

                    if (ann['bbox'][1] + ann['bbox'][3]) > self.opt.input_h:
                        ann['bbox'][3] += self.opt.input_h - (ann['bbox'][1] + ann['bbox'][3])

                    bbox, bbox_amodal, truncated = self._get_bbox_output(
                        ann['bbox'], trans_output, height, width)

                    if (bbox[2] - bbox[0]) <= 1 or (bbox[3] - bbox[1]) <= 1:
                        out_of_frame.add(ann['track_id'])
                        continue

                    box_size = ann['bbox'][2] * ann['bbox'][3]

                    if box_size < self.box_size_thresh[cls_id - 1] or last_seen_size < self.box_size_thresh[cls_id - 1]:
                        v = self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids, gt_det)
                        skipped_objs[ann['track_id']] = ann
                        continue

                    occlusion = None
                    if 'occlusion' in ann:
                        occlusion = gt_occl

                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    is_added = self._add_instance(
                            ret, gt_det, counter, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                            calib, pre_cts, track_ids, pre_vs, pre_ids, occlusion, radius_mutiplier)
                
                    if is_added:
                        counter += 1

                    ann['radius_mutiplier'] += 1
                    skipped_objs[ann['track_id']] = ann
                else:
                    radius_mutiplier = None
                    if 'radius_mutiplier' in ann:
                        radius_mutiplier = ann['radius_mutiplier']
                    occlusion = None
                    if 'occlusion' in ann:
                        occlusion = ann['occlusion']
                    if box_size < self.box_size_thresh[cls_id - 1]:
                        self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids)
                        continue

                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    self._add_instance(
                            ret, gt_det, counter, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                            calib, pre_cts, track_ids, pre_vs, pre_ids, occlusion, radius_mutiplier)
                    counter += 1

            ret['frame_id'] = img_info['frame_id']
            ret['video_id'] = img_info['video_id']
            gt_det['rw_ids'] = []
            gt_det['rw_inds'] = []

            ret['gt_det'] = gt_det
            if opt.pre_hm:
                ret['pre_hm'] = pre_hm

            meta = {
            'calib': calib,
            'c':c,
            's':s,
            'height':height,
            'width': width,
            'trans_input':trans_input,
            'trans_output':trans_output,
            'inp_height':inp.shape[1],
            'inp_width':inp.shape[2],
            'out_height': inp.shape[1] // self.opt.down_ratio,
            'out_width': inp.shape[2] // self.opt.down_ratio,
            }
            ret['meta'] = meta
            ret['image_path'] = os.path.join(self.img_dir, img_info['file_name'])
            rets.append(ret)

            pre_anns = copy.deepcopy(anns[i])
            pre_invis_anns = copy.deepcopy(invis_anns[i])

        if self.opt.random_walk and self.random_walk is None:
            from collections import OrderedDict

            id_to_idx = {}
            id_to_max_disp = {}
            ignored_tracks = []
            new_keys = []
            # For every trajectory, find and store the index of the node (box center) in which the object first becomes visible
            for i in range(len(imgs)):
                _id_to_idx, id_to_size, id_to_center, id_to_disp = self.get_idx_id_maps(rets[i], rets[i]['gt_det'])

                for k in new_keys:
                    if k not in _id_to_idx and k not in ignored_tracks:
                        id_to_idx.pop(k)

                new_keys = []
                for k in _id_to_idx.keys():
                    if k not in id_to_idx:
                        new_keys.append(k)

                # If in the first frame in which an object appears its center overlaps with the center of another object, then skip this frame to avoid confusion during initlization
                _ignored_tracks = self.check_overlap(id_to_center, new_keys)

                ignored_tracks.extend(_ignored_tracks)

                id_to_idx.update({k:v for k, v in _id_to_idx.items() if k not in id_to_idx and k not in ignored_tracks})
                for k, v in id_to_disp.items():
                    if k not in id_to_max_disp:
                        id_to_max_disp[k] = v
                    elif id_to_max_disp[k] < v:
                        id_to_max_disp[k] = v
            id_to_slot = OrderedDict(id_to_idx)
            id_to_occlen = {k:0 for k,v in id_to_slot.items()}

            valid_inds = None
            rw_down_rate = math.floor(self.opt.pool_kernel / 2) + 1
            rw_size =  (self.default_resolution[0] // (self.opt.down_ratio * rw_down_rate)) * (self.default_resolution[1] // (self.opt.down_ratio * rw_down_rate))
            # Generate supervision for the Random Walk (indices of the nodes corresponding to the object center for the frames in which an object is visible)
            for i in range(len(imgs)):
                rw_inds = np.zeros((self.max_objs), dtype=np.int64)
                rw_mask = np.zeros((self.max_objs), dtype=np.float32)
                mask_indices = np.zeros((self.max_objs, int(round(rw_size / 4))), dtype=np.int32)
                mask_weights = np.zeros((self.max_objs, int(round(rw_size / 4))), dtype=np.float32)
                rw_labels = np.zeros((self.max_objs, rw_size), dtype=np.float32)

                # Get node indices for visible objects in the current frame
                id_to_idx2, id_to_size, id_to_center, _ = self.get_idx_id_maps(rets[i], rets[i]['gt_det'])

                # There is noting to supervise in the first frame
                if i == 0:
                    for id_frame in id_to_idx2.keys():
                        seen_ids.add(id_frame)

                if i > 0:
                    rw_inds, rw_labels, rw_mask, mask_indices, mask_weights, id_to_occlen, seen_ids = self.get_rw_labels(id_to_center, id_to_size, id_to_slot, id_to_max_disp, id_to_idx2, rw_inds, rw_labels, rw_mask, mask_indices, mask_weights, rw_down_rate, id_to_occlen, seen_ids, i)

                valid_inds = rw_mask if valid_inds is None else (np.logical_or(rw_mask, valid_inds) * 1.0)

                rets[i]['rw_labels'] = rw_labels
                rets[i]['rw_mask'] = rw_mask
                rets[i]['rw_inds'] = rw_inds
                rets[i]['rw_mask_indices'] = mask_indices
                rets[i]['rw_mask_weights'] = mask_weights
                rets[i]['valid_inds'] = valid_inds

                for track_id, idx in id_to_slot.items():
                    rets[i]['gt_det']['rw_ids'].append(track_id)
                    rets[i]['gt_det']['rw_inds'].append(idx)

                rets[i]['gt_det'] = self._format_gt_det(rets[i]['gt_det'])
        else:
            for i in range(len(imgs)):
                rets[i]['gt_det'] = self._format_gt_det(rets[i]['gt_det'])
 
        return rets

    def update_pre(self, pre_anns, skipped_objs):
        updated = []
        for ann in pre_anns:
            if ann['track_id'] in skipped_objs:
                ann = skipped_objs[ann['track_id']]
            updated.append(ann)

        return updated


    def _load_data(self, index, input_len):
        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        image_info = coco.loadImgs(ids=[img_id])[0]

        video_identifier = image_info['video_id'] if not 'sensor_id' in image_info else str(image_info['video_id']) + '_' + str(image_info['sensor_id'])
        video_frames = self.video_to_images[video_identifier]
        frame_id = image_info['frame_id']

        frame_ind = self.video_to_image_map[video_identifier][img_id]
        stride = 1
        
        if self.min_frame_dist is None:
            min_frame_dist = self.opt.min_frame_dist
        else:
            min_frame_dist = self.min_frame_dist

        if self.max_frame_dist is None:
            max_frame_dist = self.opt.max_frame_dist
        else:
            max_frame_dist = self.max_frame_dist

        if 'train' in self.split and self.stride is None:
            stride = random.randint(min_frame_dist, max_frame_dist - 1)

        if  frame_ind + (input_len * stride - 1) >= len(self.video_to_image_map[video_identifier]):
            selected_images_sub = video_frames[frame_ind: len(self.video_to_image_map[video_identifier]): stride]
            selected_images = []
            while len(selected_images) < input_len:
                selected_images += selected_images_sub
                selected_images_sub.reverse()
            selected_images = selected_images[:input_len]
        elif not self.variable_stride:
            selected_images = video_frames[frame_ind: frame_ind + stride * input_len: stride]
        else:
            selected_images = []
            idx = frame_ind
            for i in range(input_len):
                selected_images.append(video_frames[idx])
                stride = random.randint(min_frame_dist, max_frame_dist- 1)
                if idx + stride >= len(video_frames):
                    idx = idx
                else:
                    idx += stride

        if not self.opt.no_temp_flip and 'train' in self.split and random.random() > 0.5:
            selected_images.reverse()

        if len(selected_images) != input_len:
            print("Query: %d; Mine: %d, seq len: %d, stride: %d" % (frame_id, selected_images[0]['frame_id'], len(video_frames), stride))

        imgs = []
        anns = []
        img_infos = []
        img_paths = []
        random_flag = random.random()
        for image_info in selected_images:
            img, ann, img_info, img_path = self._load_image_anns(image_info['id'], coco, img_dir, random_flag)
            imgs.append(img)
            anns.append(ann)
            img_infos.append(img_info)
            img_paths.append(img_path)

        return imgs, anns, img_infos, img_paths
