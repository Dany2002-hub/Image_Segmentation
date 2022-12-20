# Necessary libraries for the implementation of this algorithm
import cv2 as cv
import numpy as np
import random
# import skimage
# from skimage.segmentation import random_walker
# from skimage.data import binary_blobs


# Scaling down an image by a factor of 0.5
def down_image(x1):
    val1 = int(x1 * 0.5)
    return val1

# Function to append the pixel values of the point where the mouse has been clicked
def mouse_callback(event, x3, y1, flags, param):
    if event == 1:
        marks.append([x3, y1])
        print(marks)

def get_val(y2, x4, arr):
    if x4 < 0 or y2 < 0 or y2 >= arr.shape[0] or x4 >= arr.shape[1]:
        return np.array([-1000.0, -1000.0, -1000.0])
    else:
        return arr[y2, x4, :]

# Main function
if __name__ == "__main__":

    clicked_pixels = []

    # Colours required for segmentation
    markers = [[255, 0, 0], [255, 255, 255], [0, 0, 0]] # Blue White Black

    # User inputs
    image_datapath = "9.png"
    image = cv.imread(image_datapath)
    image_index = image_datapath[0]
    no_of_segments = int(input("Number of segments: "))
    no_of_pixels = int(str(input("Number of pixels: ")).strip())

    # Appending the clicked pixels for every segment
    for i in range(no_of_segments):
        print("SEGMENT: ", i)
        cv.imshow("Image_Display", image)
        marks = []
        cv.setMouseCallback("Image_Display", mouse_callback)
        while True:
            if len(marks) == no_of_pixels:
                break
            cv.waitKey(1)
        clicked_pixels.append(marks)
        marks = []
    print(clicked_pixels)

    # Save the pixel markings on the image
    image_copy = np.array(image)
    for i in range(no_of_segments):
        for j in range(len(clicked_pixels[i])):
            cv.circle(image_copy, (clicked_pixels[i][j][0], clicked_pixels[i][j][1]), 2, markers[i], 3)

    # Normalizing the image to increase throughput
    original_image = np.array(image)
    image = image / 255.0
    image = cv.resize(image, (int(image.shape[1] * 0.5) + 1, int(image.shape[0] * 0.5) + 1))

    marked_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)
    marked_image.fill(-1)
    segments = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)
    segments.fill(-1)
    cumulative_probability = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            direction_val = [get_val(i - 1, j, image), get_val(i, j + 1, image), get_val(i + 1, j, image), get_val(i, j - 1, image)]

            non_normalized_direction_probability = []
            for k in range(4):
                temp = np.mean(np.abs(direction_val[k] - image[i, j, :]))
                temp = np.exp(-1 * np.power(temp, 2))
                non_normalized_direction_probability.append(temp)

            non_normalized_direction_probability = np.array(non_normalized_direction_probability)
            normalized_direction_probability = non_normalized_direction_probability / np.sum(non_normalized_direction_probability)
            cumulative_probability[i, j, 0] = normalized_direction_probability[0]

            for k in range(1, 4):
                cumulative_probability[i, j, k] = cumulative_probability[i, j, k - 1] + normalized_direction_probability[k]

    for i in range(no_of_segments):
        for j in range(len(clicked_pixels[i])):
            marked_image[down_image(clicked_pixels[i][j][1]), down_image(clicked_pixels[i][j][0])] = i
            segments[down_image(clicked_pixels[i][j][1]), down_image(clicked_pixels[i][j][0])] = i

    # Random Walks Algorithm
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            if segments[i][j] == -1:
                x = i
                y = j
                while marked_image[x, y] == -1:
                    random_var = random.random()
                    if cumulative_probability[x, y, 0] > random_var:
                        x -= 1
                    elif cumulative_probability[x, y, 1] > random_var:
                        y += 1
                    elif cumulative_probability[x, y, 2] > random_var:
                        x += 1
                    else:
                        y -= 1

                segments[i, j] = marked_image[x, y]

    print("Marking Completed")

    # def built_in_random_walker():
    #     img = cv.imread(image_datapath)
    #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     img = img / 255.0
    #
    #     markers_1 = np.zeros(img.shape, dtype=np.uint)
    #     for i in range(no_of_segments):
    #         for j in range(len(clicked_pixels[i])):
    #             x_c = clicked_pixels[i][j][0]
    #             y_c = clicked_pixels[i][j][1]
    #             markers_1[y_c, x_c] = i + 1
    #
    #     labels = random_walker(img, markers_1, beta=10, mode='bf')
    #     output_img = np.array(original_image)
    #
    #     for i in range(output_img.shape[0]):
    #         for j in range(output_img.shape[1]):
    #             output_img[i, j] = markers[labels[i, j]]
    #
    #     cv.imwrite("in_built_img_" + image_index + ".png", output_img)


    # Output image by Random Walks Algorithm
    output_image = np.array(original_image)
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = markers[segments[down_image(i), down_image(j)]]

    # Writing the image to the files
    cv.imwrite("concatenated_img_" + image_index + ".png", np.concatenate((image_copy, output_image), axis=1))
