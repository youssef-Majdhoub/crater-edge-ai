import numpy as np
import matplotlib.pyplot as plt

loaded_image = np.load("image.npy")


# since we have made the image in 4 quadrants,
# each only needs 6 bit to encode, si lossless compression is possible
def lossless_compression(image):
    # Get the shape of the image
    height, width = image.shape
    # Create an empty array to hold the compressed data
    compressed_data = np.zeros((height, width), dtype=np.uint8)
    zones = [
        (0, 0, 100, 100),
        (0, 100, 100, 200),
        (100, 100, 200, 200),
        (100, 0, 200, 100),
    ]
    for i in range(len(zones)):
        zone = zones[i]
        compressed_data[zone[0] : zone[2], zone[1] : zone[3]] = (
            image[zone[0] : zone[2], zone[1] : zone[3]] - i * 64
        ).astype(np.uint8)
    # now we need to encode the compressed data into 6 bits,
    # we can do this by packing 10 pixels into uint64, since 10 * 6 = 60 bits,
    # which fits into 64 bits
    result = np.zeros((height * width // 10,), dtype=np.uint64)
    for i in range(0, height * width, 10):
        # get the next 10 pixels
        pixels = compressed_data.flatten()[i : i + 10]
        # pack them into a uint64
        packed = 0
        for j in range(len(pixels)):
            packed |= int(pixels[j]) << (j * 6)
        result[i // 10] = packed
    return result


def lossless_decompression(compressed_data, height, width):
    # Create an empty array to hold the decompressed data
    decompressed_data = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(compressed_data)):
        # unpack the uint64 into 10 pixels
        packed = compressed_data[i]
        for j in range(10):
            pixel = (packed >> (j * 6)) & 63
            decompressed_data[(i * 10 + j) // width, (i * 10 + j) % width] = pixel
    # now we need to add back the zone offsets
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


def save_compressed_data(compressed_data, width, height, filename):
    # Save the compressed data to a file
    np.savez(filename, compressed_data=compressed_data, width=width, height=height)


def load_decompress_compare(filename):
    # Load the compressed data from a file
    data = np.load(filename)
    compressed_data = data["compressed_data"]
    width = data["width"]
    height = data["height"]
    # Decompress the data
    decompressed_data = lossless_decompression(compressed_data, height, width)
    # Compare the decompressed data with the original image
    if np.array_equal(decompressed_data, loaded_image):
        print("Decompressed data matches the original image.")
    else:
        print("Decompressed data does not match the original image.")
    compare_size(compressed_data, loaded_image)
    return decompressed_data


def compare_size(compressed_data, original_image):
    compressed_size = compressed_data.nbytes
    original_size = original_image.nbytes
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Original size: {original_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}")


if __name__ == "__main__":
    # show the original image
    compressed_data = lossless_compression(loaded_image)
    save_compressed_data(
        compressed_data,
        loaded_image.shape[1],
        loaded_image.shape[0],
        "compressed_data.npz",
    )
    plt.subplot(1, 2, 1)
    plt.imshow(loaded_image, cmap="gray")
    plt.title("Original Image")
    # show the decompressed image
    decompressed_data = load_decompress_compare("compressed_data.npz")
    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_data, cmap="gray")
    plt.title("Decompressed Image")
    plt.show()
