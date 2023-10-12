import copy
import torch
import numpy as np
from matplotlib import cm
import os
import cv2
from PIL import Image
from torchvision.transforms import ToPILImage


def get_relevance_map(relevance_map, size, positive=True):
    relevance_map = copy.deepcopy(relevance_map)
    if positive:
        cmap = cm.get_cmap('hot')
    else:
        cmap = cm.get_cmap('cool')
    relevance_map =  relevance_map.numpy().copy()
    relevance_map = relevance_map.reshape(relevance_map.shape[1:])
    if not positive:
        relevance_map = relevance_map * -1.
    relevance_map[relevance_map[:,:] <= 0.] = 0.
    denominator = (np.max(relevance_map) - np.min(relevance_map))
    if denominator < 0.000001:
        relevance_map = relevance_map
    else:
        relevance_map = (relevance_map - np.min(relevance_map)) / denominator
    relevance_map = cmap(relevance_map)
    relevance_map = np.uint8(relevance_map * 255)
    relevance_map[:,:,3] = relevance_map[:,:,0]
    relevance_map[:,:,3] = np.ceil(relevance_map[:,:,3] * 0.5)
    relevance_map = Image.fromarray(relevance_map, mode='RGBA').resize(size, resample=Image.NEAREST)
    return relevance_map


def superimpose_relevance_map(relevance_maps, background_image):
    combined_image = copy.deepcopy(background_image)
    for relevance_map in relevance_maps:
        combined_image.paste(relevance_map, (0, 0), relevance_map)
    return combined_image


def create_image_sequence(original_image_tensor,
                          relevances,
                          size,
                          include_negative=True):
    image_list = []

    tensor_image_transform = ToPILImage()
    original_image = tensor_image_transform(original_image_tensor)
    original_image = original_image.resize(size, resample=Image.NEAREST)
    original_image = original_image.convert(mode='RGBA')

    for relvances_i in relevances:
        relevance_maps = [get_relevance_map(relevance_map=relvances_i, size=size, positive=True)]
        if include_negative:
            relevance_maps.append(get_relevance_map(relevance_map=relvances_i, size=size, positive=False))
        image_list.append(superimpose_relevance_map(relevance_maps=relevance_maps, background_image=original_image))

    return image_list


def save_images_as_gif(image_list, save_name):
    img_gen = (img for img in image_list)
    img = next(img_gen)
    img.save(fp=os.path.join('results', save_name + '.gif'),
             format='GIF',
             append_images=img_gen,
             save_all=True,
             duration=2/1000,
             loop=0)


def save_images_as_video(image_list, save_name):
    size = image_list[0].size
    save_path = os.path.join('results', save_name + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(save_path, fourcc, 60, size)

    for i in range(len(image_list)):
        img = image_list[i]
        img = img.convert('RGB')
        frame = np.asarray(img)
        new_frame = frame.copy()
        new_frame[:,:,0] = frame[:,:,2]
        new_frame[:,:,2] = frame[:,:,0]
        out.write(new_frame)
    out.release()


if __name__ == "__main__":
    pass
