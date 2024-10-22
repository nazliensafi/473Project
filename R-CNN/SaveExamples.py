import matplotlib.pyplot as plt
import cv2
import os
import re
import constants

# The number of the samples that are of interest
samples = {
    "normal": ["122", "129"],
    "benign": ["350", "356", "360", "371"],
    "malignant": ["169", "170"],
}

if __name__ == "__main__":
    for sample_type in samples.keys():

        for sample_number in samples[sample_type]:
            image_path = os.path.join(
                constants.TESTING_PATH,
                sample_type,
                f"{sample_type} ({sample_number}).png",
            )
            mask_path = re.sub("\.png", "_mask.png", image_path)
            segmentation_path = re.sub("\.png", "_segmentation.png", image_path)

            # Load the three images
            img1 = cv2.imread(image_path)
            img2 = cv2.imread(mask_path)
            img3 = cv2.imread(segmentation_path)

            # # # Resize the images to a height and width of 256 pixels
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

            img1_resized = cv2.resize(img1_rgb, (256, 256))
            img2_resized = cv2.resize(img2_rgb, (256, 256))
            img3_resized = cv2.resize(img3_rgb, (256, 256))

            # Save the resulting image
            fig, ax = plt.subplots(1, 3, figsize=(7, 2))

            # Display the three images side by side
            ax[0].imshow(img1_resized)
            ax[1].imshow(img2_resized)
            ax[2].imshow(img3_resized)

            destination = os.path.join(
                constants.DATA_PATH, "Examples", f"{sample_type} ({sample_number}).png"
            )

            plt.savefig(destination)
