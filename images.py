import random

from PIL import Image, ImageDraw

from main import helper

images = list(map(Image.open, ['data/1/DSC03230.png', 'data/1/DSC03240.png']))
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]

pairs = helper()
print(len(pairs))

draw = ImageDraw.Draw(new_im)
for i in range(len(pairs)):
    x1 = pairs[i][0].x
    y1 = pairs[i][0].y
    x2 = pairs[i][1].x + images[0].size[0]
    y2 = pairs[i][1].y
    draw.line((x1, y1, x2, y2), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

new_im.save('data/1/result.png')
