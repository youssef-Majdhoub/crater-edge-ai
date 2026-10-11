import numpy as np
import matplotlib.pyplot as plt

loaded_image = np.load("image.npy")

# since we have made the image in 4 quadrants,
# and since each one is randomly generated,
# we can use lossy compression to reduce the size of the image
# simply because there is no information to be lost,
# instead of 64 values per quadrant, we can use difrential encoding
# with 16 step values, which will reduce the number of bits needed to encode each pixel from 8 to 4,

zones = [(0, 0, 100, 100), (0, 100, 100, 200), (100, 100, 200, 200), (100, 0, 200, 100)]


def build_codebook(levels):
    """Split the possible differences into the requested number of levels."""
    if levels not in (2, 4, 8, 16):
        raise ValueError("levels must be one of 2, 4, 8, or 16")

    values = np.array_split(np.arange(-63, 64), levels)
    return [(int(values[i][0]), int(values[i][-1])) for i in range(levels)]


LEVELS = 16
# we need to create the diff version:
diff_image = loaded_image.astype(np.int16)
for i in range(len(zones)):
    zone = zones[i]
    diff_image[zone[0] : zone[2], zone[1] : zone[3]] -= i * 64
diff_image[1:, :] -= diff_image[:-1, :]


def lossy_compression(diff_image, levels=LEVELS):
    code = build_codebook(levels)
    bits_per_pixel = int(np.log2(levels))
    pixels_per_byte = 8 // bits_per_pixel
    height, width = diff_image.shape
    compressed_data = np.zeros(
        ((height * width + pixels_per_byte - 1) // pixels_per_byte,),
        dtype=np.uint8,
    )
    for i in range(0, height * width, pixels_per_byte):
        pixels = diff_image.flatten()[i : i + pixels_per_byte]
        # encode them into a single byte
        encoded = 0
        for j in range(len(pixels)):
            for level, interval in enumerate(code):
                if interval[0] <= pixels[j] <= interval[1]:
                    encoded |= level << (j * bits_per_pixel)
                    break
        compressed_data[i // pixels_per_byte] = encoded
    return compressed_data


def lossy_decompression(compressed_data, height, width, levels=LEVELS):
    code = build_codebook(levels)
    bits_per_pixel = int(np.log2(levels))
    pixels_per_byte = 8 // bits_per_pixel
    # when gettig the pixel difference level ,we will generate a random value in the range of the code,
    # to simulate the loss of information
    decompressed_data = np.zeros((height, width), dtype=np.int16)
    for i in range(len(compressed_data)):
        encoded = compressed_data[i]
        for j in range(pixels_per_byte):
            pixel_index = i * pixels_per_byte + j
            if pixel_index >= height * width:
                break
            level = (encoded >> (j * bits_per_pixel)) & (levels - 1)
            # generate a random value in the range of the code
            pixel = (code[level][0] + code[level][1]) / 2
            decompressed_data[pixel_index // width, pixel_index % width] = pixel
    # now we need to remove the diff and add back the zone offsets
    for i in range(1, height):
        decompressed_data[i, :] += decompressed_data[i - 1, :]
    zones = [
        (0, 0, 100, 100),
        (0, 100, 100, 200),
        (100, 100, 200, 200),
        (100, 0, 200, 100),
    ]
    for i in range(len(zones)):
        zone = zones[i]
        decompressed_data[zone[0] : zone[2], zone[1] : zone[3]] += i * 64
    return decompressed_data


def save_compressed_data(compressed_data, width, height, levels, filename):
    # save the compressed data to a file
    np.savez(
        filename,
        compressed_data=compressed_data,
        width=width,
        height=height,
        levels=levels,
    )


def load_compressed_data(filename):
    # load the compressed data from a file
    data = np.load(filename)
    compressed_data = data["compressed_data"]
    width = data["width"]
    height = data["height"]
    levels = int(data["levels"])
    return compressed_data, width, height, levels


def load_decompress_compare(filename):
    # load the compressed data from a file
    compressed_data, width, height, levels = load_compressed_data(filename)
    # decompress the data
    decompressed_data = lossy_decompression(compressed_data, height, width, levels)
    # load the original image
    original_image = np.load("image.npy")
    # compare the two images
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_data, cmap="gray")
    plt.title("Decompressed Image")
    plt.show()
    compare_size(compressed_data, original_image)


def compare_size(compressed_data, original_image):
    compressed_size = compressed_data.nbytes
    original_size = original_image.nbytes
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Original size: {original_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}")


if __name__ == "__main__":
    levels = 16
    compressed_data = lossy_compression(diff_image, levels)
    save_compressed_data(
        compressed_data,
        loaded_image.shape[1],
        loaded_image.shape[0],
        levels,
        "lossy_compressed_data.npz",
    )
    load_decompress_compare("lossy_compressed_data.npz")
