import os
import cv2
import numpy as np

def crop_img_func(img, img_name_ori, mask=None, jpg_dct=None, crop_size=512):
    # img, mask, jpg_dct all in shape [H, W, C]
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

    h, w, c = img.shape
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
            if use_jpg_dct:
                crop_jpe_dct = jpg_dct[h-crop_size:h,x1:x2]
                crop_jpe_dcts['%s_%s_%s'%(img_name, h_grids, w_idx)] = crop_jpe_dct
            if use_mask:
                crop_mask = mask[h-crop_size:h,x1:x2]
                crop_masks['%s_%s_%s'%(img_name, h_grids, w_idx)] = crop_mask

    if (w%crop_size!=0) and (h%crop_size!=0):
        crop_imgs['%s_%s_%s'%(img_name, h_grids, w_grids)] = img[h-crop_size:h,w-crop_size:w]
        if use_jpg_dct:
            crop_jpe_dct = jpg_dct[h-crop_size:h,w-crop_size:w]
            crop_jpe_dcts['%s_%s_%s'%(img_name, h_grids, w_grids)] = crop_jpe_dct
        if use_mask:
            crop_mask = mask[h-crop_size:h,w-crop_size:w]
            crop_masks['%s_%s_%s'%(img_name, h_grids, w_grids)] = crop_mask

    h, w = img.shape[:2]
    meta_info = {'h': h, 'w': w, 'h_grids': h_grids, 'w_grids': w_grids, 'crop_size': crop_size, 'suffix': suffix, 'img_name': img_name, 'channels': channels}

    if use_mask and use_jpg_dct:
        return crop_imgs, meta_info, crop_masks, crop_jpe_dcts
    elif use_mask:
        return crop_imgs, meta_info, crop_masks
    elif use_jpg_dct:
        return crop_imgs, meta_info, crop_jpe_dcts
    else:
        return crop_imgs, meta_info


def combine_img_func(crop_imgs, meta_info):
    img_name = meta_info['img_name']
    img_h = meta_info['h']
    img_w = meta_info['w']
    channels = meta_info['channels']
    h_grids = meta_info['h_grids']
    w_grids = meta_info['w_grids']
    crop_size = meta_info['crop_size']

    re_img = np.zeros((img_h, img_w)) if (channels==1) else np.zeros((img_h, img_w, 3))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2, x1:x2] = crop_imgs['%s_%s_%s'%(img_name, h_idx, w_idx)]

    if w_grids*crop_size<img_w:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2,img_w-crop_size:img_w] = crop_imgs['%s_%s_%s'%(img_name, h_idx, w_grids)]

    if h_grids*crop_size<img_h:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            re_img[img_h-crop_size:img_h,x1:x2] = crop_imgs['%s_%s_%s'%(img_name, h_grids, w_idx)]

    if w_grids*crop_size<img_w and h_grids*crop_size<img_h:
        re_img[img_h-crop_size:img_h,img_w-crop_size:img_w] = crop_imgs['%s_%s_%s'%(img_name, h_grids, w_grids)]

    return re_img


if __name__=="__main__":
    ### Crop size
    crop_size = 256

    ### Make dirs for demo
    if not os.path.exists('demo_img_dir'):
        os.makedirs('demo_img_dir')
    if not os.path.exists('demo_mask_dir'):
        os.makedirs('demo_mask_dir')

    ### Read demo image and mask
    input_img_path = 'img_demo.png'
    input_mask_path = 'mask_demo.png'

    img = cv2.imread(input_img_path)
    mask = cv2.imread(input_mask_path)

    ### Crop image and mask
    crop_imgs, meta_info, crop_masks = crop_img_func(img=img, img_name_ori=input_img_path, mask=mask, crop_size=crop_size)

    ### Save the cropped image and mask
    for k,v in crop_imgs.items():
        cv2.imwrite('demo_img_dir/'+k+'.jpg', v)
    for k,v in crop_masks.items():
        cv2.imwrite('demo_mask_dir/'+k+'.png', v)

    ### Reconstruction from variable
    rec_img = combine_img_func(crop_imgs, meta_info)
    cv2.imwrite('rec_img1.png', rec_img)

    rec_mask = combine_img_func(crop_imgs, meta_info)
    cv2.imwrite('rec_mask1.png', rec_mask)

    ### Reconstruction from files
    collected_imgs = {k.split('.')[0]: cv2.imread('demo_img_dir/'+k) for k in os.listdir('demo_img_dir')}
    collected_masks = {k.split('.')[0]: cv2.imread('demo_mask_dir/'+k) for k in os.listdir('demo_mask_dir')}

    rec_img = combine_img_func(collected_imgs, meta_info)
    cv2.imwrite('rec_img2.png', rec_img)

    rec_mask = combine_img_func(collected_masks, meta_info)
    cv2.imwrite('rec_mask2.png', rec_mask)
