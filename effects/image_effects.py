from PIL import Image, \
                ImageDraw, \
                ImageOps, \
                ImageFilter

from typing import List, Any

import math
import random

def apply_twirl(
    input_image: Image.Image,
    angle: float,
    radius: float,
) -> Image.Image:
    if radius == 0:
        return input_image

    output_image = input_image.copy()

    width, height = output_image.size

    draw = ImageDraw.Draw(output_image)
    cx, cy = width // 2, height // 2

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy

            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance <= radius:
                angle_offset = angle * (1 - distance / radius)
                theta = math.atan2(dy, dx) + angle_offset

                tx = int(cx + distance * math.cos(theta))
                ty = int(cy + distance * math.sin(theta))

                if 0 <= tx < width and 0 <= ty < height:
                    pixel = output_image.getpixel((tx, ty))
                    draw.point((x, y), pixel)

    return output_image

def apply_skew(
    input_image: Image.Image,
    angle: float,
    factor: float,
) -> Image.Image:
    if factor == 0.0:
        return input_image

    width, height = input_image.size

    output_image = Image.new(
        mode=input_image.mode,
        size=(width, height),
        color=(0, 0, 0),
    )

    matrix = [1, factor * math.tan(math.radians(angle)), 0, 0, 1, 0]

    output_image = input_image.transform(
        size=input_image.size,
        method=Image.AFFINE,
        data=matrix,
        resample=Image.BICUBIC,
    )

    return output_image

def apply_pixelate(input_image: Image.Image, block_size: int) -> Image.Image:
    if block_size == 0:
        return input_image

    output_image = input_image.copy()

    width, height = output_image.size

    for x in range(0, width, block_size):
        for y in range(0, height, block_size):
            box = (x, y, x + block_size, y + block_size)
            average_color = input_image.crop(box).resize((1, 1), resample=Image.BOX).getpixel((0, 0))
            output_image.paste(average_color, box)

    return output_image

def apply_threshold(input_image: Image.Image, value: int) -> Image.Image:
    if value == 0:
        return input_image

    output_image = ImageOps.grayscale(input_image)
    output_image = output_image.point(lambda pixel: pixel > value and 255)

    return output_image.convert("RGB")

def apply_blur(input_image: Image.Image, radius: float) -> Image.Image:
    if radius == 0.0:
        return input_image

    output_image = input_image.filter(ImageFilter.GaussianBlur(radius))

    return output_image

def apply_rotation(input_image: Image.Image, angle: float) -> Image.Image:
    output_image = input_image.copy()
    output_image = output_image.rotate(angle)

    return output_image

def apply_scale(
    input_image: Image.Image,
    horizontal_factor: float,
    vertical_factor: float,
) -> Image.Image:
    width_input, height_input = input_image.size

    output_image = Image.new(
        mode=input_image.mode,
        size=(width_input, height_input),
        color=(0, 0, 0),
    )

    width_output = int(width_input * horizontal_factor)
    height_output = int(height_input * vertical_factor)

    scaled_content = input_image.resize((width_output, height_output), Image.LANCZOS)

    paste_position = ((width_input - width_output) // 2, (height_input - height_output) // 2)

    output_image.paste(scaled_content, paste_position)

    return output_image

def apply_offset(
    input_image: Image.Image,
    value_x: int,
    value_y: int,
) -> Image.Image:
    width, height = input_image.size

    output_image = Image.new(
        mode=input_image.mode,
        size=(width, height),
        color=(0, 0, 0),
    )

    output_image.paste(input_image, (value_x, value_y))

    return output_image

def apply_shuffle(
    input_image: Image.Image,
    tile_size_x: int,
    tile_size_y: int,
    strength: float,
) -> Image.Image:
    width, height = input_image.size

    tiles_amount_x = width // tile_size_x
    tiles_amount_y = height // tile_size_y

    tile_coordinates = [(x, y) for x in range(tiles_amount_x) for y in range(tiles_amount_y)]

    shuffle_with_strength(tile_coordinates, strength)

    output_image = Image.new(
        mode=input_image.mode,
        size=(width, height),
        color=(0, 0, 0),
    )

    for x in range(tiles_amount_x):
        for y in range(tiles_amount_y):
            original_x, original_y = tile_coordinates.pop(0)

            tile = input_image.crop((
                original_x * tile_size_x,
                original_y * tile_size_y,
                (original_x + 1) * tile_size_x,
                (original_y + 1) * tile_size_y
            ))

            output_image.paste(
                tile,
                (x * tile_size_x, y * tile_size_y),
            )

    return output_image

def shuffle_with_strength(data: List[Any], strength: float) -> List[Any]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Strength parameter must be between 0.0 and 1.0 inclusive.")

    if strength == 0.0:
        return data

    if strength == 1.0:
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        return shuffled_data

    data_length = len(data)

    num_shuffled_elements = int(strength * data_length)
    shuffled_indices = list(range(data_length))

    random.shuffle(shuffled_indices)

    for i in range(num_shuffled_elements):
        j = shuffled_indices[i]
        data[i], data[j] = data[j], data[i]

    return data

def apply_waves(
    input_image: Image.Image,
    length_horizontal: float,
    amplitude_horizontal: float,
    shift_horizontal: float,
    length_vertical: float,
    amplitude_vertical: float,
    shift_vertical: float,
) -> Image.Image:
    output_image = input_image.copy()

    width, height = output_image.size

    for y in range(height):
        for x in range(width):
            offset_x = int(amplitude_horizontal * math.sin(2 * math.pi * x / length_horizontal + shift_horizontal))

            new_x = x + offset_x

            if 0 <= new_x < width:
                pixel = input_image.getpixel((new_x, y))
                output_image.putpixel((x, y), pixel)

    for x in range(width):
        for y in range(height):
            offset_y = int(amplitude_vertical * math.sin(2 * math.pi * y / length_vertical + shift_vertical))

            new_y = y + offset_y

            if 0 <= new_y < height:
                pixel = input_image.getpixel((x, new_y))
                output_image.putpixel((x, y), pixel)

    return output_image
