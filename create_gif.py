import os
import cv2
import imageio.v2 as imageio


def natural_sort_key(name: str):
    prefix = name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else prefix


def resize_to_width(img, target_width):
    if img.shape[1] == target_width:
        return img

    scale = target_width / img.shape[1]
    new_height = max(1, int(img.shape[0] * scale))
    return cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)


def add_text_overlay(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6.0
    thickness = 8
    padding = 10

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = 1100
    y = 800

    # cv2.rectangle(
    #     img,
    #     (x - padding, y - text_h - padding),
    #     (x + text_w + padding, y + baseline + padding),
    #     (0, 0, 0),
    #     -1,
    # )
    cv2.putText(img, text, (x, y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    return img

classes = []
images = []
imaginators = []

for filename in os.listdir("imgs"):
    if filename.endswith(".png"):
        filename = filename.split(".")[0]
        class_name = filename.split("_")[-1]
        if class_name not in classes:
            classes.append(class_name)
        # img = cv2.imread(os.path.join("imgs", filename + ".png"))
        if filename.split("_")[0].lower() == "imaginator":
            imaginators.append(filename)
        else:
            images.append(filename)
            
images.sort(key=natural_sort_key)
imaginators.sort(key=lambda x: natural_sort_key(x.replace("imaginator_", "", 1)))

print(f"Found {len(images)} images and {len(imaginators)} imaginators and {len(classes)} classes.")

imaginator_lookup = {
    name.replace("imaginator_", "", 1): name for name in imaginators
}

frames = []
for image_name in images:
    imaginator_name = imaginator_lookup.get(image_name)
    if imaginator_name is None:
        print(f"Skipping {image_name}: imaginator not found.")
        continue

    class_name = image_name.split("_", 1)[1] if "_" in image_name else image_name

    image_path = os.path.join("imgs", image_name + ".png")
    imaginator_path = os.path.join("imgs", imaginator_name + ".png")

    image = cv2.imread(image_path)
    imaginator = cv2.imread(imaginator_path)

    if image is None:
        print(f"Skipping {image_name}: could not read {image_path}")
        continue
    if imaginator is None:
        print(f"Skipping {image_name}: could not read {imaginator_path}")
        continue

    imaginator = resize_to_width(imaginator, image.shape[1])
    stacked = cv2.vconcat([image, imaginator])
    stacked = add_text_overlay(stacked, class_name)

    # Convert BGR to RGB for GIF encoding.
    frames.append(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))

if not frames:
    raise RuntimeError("No frames were created. Check image files and naming.")

output_path = os.path.join("imgs", "stacked_stimuli.gif")
imageio.mimsave(output_path, frames, duration=1.2, loop=0)

print(f"Saved GIF to {output_path} with {len(frames)} frames.")
