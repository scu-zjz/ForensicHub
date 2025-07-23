import os
import cv2
import imagesize
import numpy as np
from tqdm import tqdm

thres = 64*256

def getdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_img_func(img, img_name_ori, mask=None, jpg_dct=None, crop_size=512):
    # img is the RGB images loaded through the cv2.imread function, img = cv2.imread('xxx.jpg'). Type: np.array, shape: [H, W, 3].
    # img_name_ori is the image's name. For example, for the image in path 'OSTF/OSTF_train/images/srnet_102.jpg', its image_name_ori is 'srnet_102.jpg'. Type: string.
    # mask (optional) is the binary mask indicating the tampered region, it is loaded through the cv2.imread function, mask = cv2.imread('xxx.png', 0). Type: np.array, shape: [H, W, C].
    # jpg_dct (optional) is the DCT_coff loaded through the jpegio.read function, dct = jpegio.read('xxx.jpg').coef_arrays[0].copy(). Type: np.array, shape: [H, W, 1].
    # crop_size is the target size of cropped image patches. Type: int.
    if '.' in img_name_ori:
        img_name = '.'.join(img_name_ori.split('.')[:-1])
        suffix = img_name_ori.split('.')[-1]
    else:
        suffix = ""
    if mask is None:
        use_mask=False
    else:
        use_mask=True
        crop_masks = {}

    if jpg_dct is None:
        use_jpg_dct=False
    else:
        use_jpg_dct=True
        crop_jpe_dcts = {}

    h, w = img.shape[:2]
    if use_mask:
        hm, wm = mask.shape[:2]
        if (hm!=h) or (wm!=w):
            mask = cv2.resize(mask, (w, h), 0)
    if min(h, w)<512:
        if h>w:
            new_h = int(h*512/w+0.5)
            new_w = 512
        else:
            new_w = int(w*512/h+0.5)
            new_h = 512
        img = cv2.resize(img, (new_w, new_h))
        if use_mask:
            mask = cv2.resize(mask, (new_w, new_h))
    h, w, c = img.shape
    assert min(h, w)>=512

    if min(h, w)<512: ###
        pad_img_v = 255 if img_name_ori[0]=='X' else 0
        if w>h:
            new_img = np.full(shape=(512, w, 3), fill_value=pad_img_v, dtype=np.uint8)
            new_img[:h, :w] = img
            img = new_img
            if use_mask:
                new_mask = np.full(shape=(512, w), fill_value=0, dtype=np.uint8)
                new_mask[:h, :w] = mask
                mask = new_mask
        else:
            new_img = np.full(shape=(h, 512, 3), fill_value=pad_img_v, dtype=np.uint8)
            new_img[:h, :w] = img
            img = new_img
            if use_mask:
                new_mask = np.full(shape=(h, 512), fill_value=0, dtype=np.uint8)
                new_mask[:h, :w] = mask
                mask = new_mask

    channels = 3 if (len(img.shape)==3) else 1
    h_grids = h // crop_size
    w_grids = w // crop_size

    crop_imgs = {}

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_img = img[y1:y2, x1:x2, :]
            crop_imgs['%s_%s_%s'%(img_name, h_idx, w_idx)] = crop_img
            if use_jpg_dct:
                crop_jpe_dct = jpg_dct[y1:y2, x1:x2]
                crop_jpe_dcts['%s_%s_%s'%(img_name, h_idx, w_idx)] = crop_jpe_dct
            if use_mask:
                crop_mask = mask[y1:y2, x1:x2]
                crop_masks['%s_%s_%s'%(img_name, h_idx, w_idx)] = crop_mask

    if w%crop_size!=0:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_imgs['%s_%s_%s'%(img_name, h_idx, w_grids)] = img[y1:y2,w-crop_size:w]
            # print(img.shape, crop_imgs['%s_%s_%s'%(img_name, h_idx, w_grids)].shape)
            if use_jpg_dct:
                crop_jpe_dct = jpg_dct[y1:y2,w-crop_size:w]
                crop_jpe_dcts['%s_%s_%s'%(img_name, h_idx, w_grids)] = crop_jpe_dct
            if use_mask:
                crop_mask = mask[y1:y2,w-crop_size:w]
                crop_masks['%s_%s_%s'%(img_name, h_idx, w_grids)] = crop_mask

    if h%crop_size!=0:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            crop_imgs['%s_%s_%s'%(img_name, h_grids, w_idx)] = img[h-crop_size:h,x1:x2]
            # print(img.shape, crop_imgs['%s_%s_%s'%(img_name, h_grids, w_idx)].shape)
            if use_jpg_dct:
                crop_jpe_dct = jpg_dct[h-crop_size:h,x1:x2]
                crop_jpe_dcts['%s_%s_%s'%(img_name, h_grids, w_idx)] = crop_jpe_dct
            if use_mask:
                crop_mask = mask[h-crop_size:h,x1:x2]
                crop_masks['%s_%s_%s'%(img_name, h_grids, w_idx)] = crop_mask

    if (w%crop_size!=0) and (h%crop_size!=0):
        crop_imgs['%s_%s_%s'%(img_name, h_grids, w_grids)] = img[h-crop_size:h,w-crop_size:w]
        # ch, cw = crop_imgs['%s_%s_%s'%(img_name, h_grids, w_grids)].shape[:2]
        if use_jpg_dct:
            crop_jpe_dct = jpg_dct[h-crop_size:h,w-crop_size:w]
            crop_jpe_dcts['%s_%s_%s'%(img_name, h_grids, w_grids)] = crop_jpe_dct
        if use_mask:
            crop_mask = mask[h-crop_size:h,w-crop_size:w]
            crop_masks['%s_%s_%s'%(img_name, h_grids, w_grids)] = crop_mask

    h, w = img.shape[:2]
    meta_info = {'h': h, 'w': w, 'h_grids': h_grids, 'w_grids': w_grids, 'crop_size': crop_size, 'suffix': suffix, 'img_name': img_name, 'channels': channels}

    for k,v in crop_imgs.items():
        ch, cw = v.shape[:2]
        if (ch!=512) or (cw!=512):
            print(k, ch, cw)

    if use_mask and use_jpg_dct:
        return crop_imgs, meta_info, crop_masks, crop_jpe_dcts
    elif use_mask:
        return crop_imgs, meta_info, crop_masks
    elif use_jpg_dct:
        return crop_imgs, meta_info, crop_jpe_dcts
    else:
        return crop_imgs, meta_info

# img = cv2.imread('OSTF/OSTF_train/images/srnet_102.jpg')
# mask = cv2.imread('OSTF/OSTF_train/images/srnet_102.jpg', 0)
# crop_imgs, meta_info, crop_masks = crop_img_func(img=img, img_name_ori='srnet_102.jpg', mask=mask)

for nm in ('T-SROIE', 'OSTF', 'Tampered-IC13'):
    for tnm in ('_train', '_test'):
        alls_img_root = os.path.join('cutted_datasets_alls', nm+tnm, 'images')
        alls_msk_root = os.path.join('cutted_datasets_alls', nm+tnm, 'masks')
        fake_img_root = os.path.join('cutted_datasets_fakes', nm+tnm, 'images')
        fake_msk_root = os.path.join('cutted_datasets_fakes', nm+tnm, 'masks')
        getdir(alls_img_root)
        getdir(alls_msk_root)
        getdir(fake_img_root)
        getdir(fake_msk_root)
        alls_img_ori = os.path.join(nm, nm+tnm, 'images')
        alls_msk_ori = os.path.join(nm, nm+tnm, 'masks')
        fake_img_ori = os.path.join(nm, nm+tnm, 'images')
        fake_msk_ori = os.path.join(nm, nm+tnm, 'masks')
        for x in tqdm(os.listdir(alls_img_ori)):
            img = cv2.imread(os.path.join(alls_img_ori, x))
            msk = cv2.imread(os.path.join(alls_msk_ori, x[:-4]+'.png'), 0)
            crop_imgs, meta_info, crop_masks = crop_img_func(img=img, img_name_ori=x, mask=msk)
            for k,v in crop_imgs.items():
                mskv = crop_masks[k]
                msum = mskv.sum()
                if (msum==0):
                    cv2.imwrite(os.path.join(alls_img_root, k+'.jpg'), v)
                    cv2.imwrite(os.path.join(alls_msk_root, k+'.png'), mskv)
                elif msum>=thres:
                    cv2.imwrite(os.path.join(fake_img_root, k+'.jpg'), v)
                    cv2.imwrite(os.path.join(fake_msk_root, k+'.png'), mskv)
                    cv2.imwrite(os.path.join(alls_img_root, k+'.jpg'), v)
                    cv2.imwrite(os.path.join(alls_msk_root, k+'.png'), mskv)

