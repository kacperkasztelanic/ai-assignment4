import os

import numpy as np
from PIL import Image, ImageDraw


def print_image(image_paths, pairs):
    images = list(map(Image.open, image_paths))
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    x_offset = images[0].size[0]

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (x_offset, 0))

    draw = ImageDraw.Draw(new_im)
    for pair in pairs:
        x1 = pair[0].coords[0]
        y1 = pair[0].coords[1]
        x2 = pair[1].coords[0] + x_offset
        y2 = pair[1].coords[1]
        color = tuple(np.random.randint(256, size=3))
        draw.line((x1, y1, x2, y2), fill=color)
    result_path = os.path.join(os.path.dirname(image_paths[0]), 'result.png')
    new_im.save(result_path)
