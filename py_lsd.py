from pathlib import Path

import cv2 as cv
import numpy as np

import lsd_ext


def draw_lines(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    lines = lines.reshape(-1, 2, 2)
    lines = lines.astype(int)
    for p, q in lines:
        p = tuple(p)
        q = tuple(q)
        cv.line(image, p, q, (0, 255, 0), 2)
    return image

def process(image_dir: str) -> None:
    """Save predicted lines into .npy files"""
    for image_path in Path(image_dir).glob("*"):
        image = cv.imread(str(image_path))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        output = lsd_ext.lsd(gray)
        lines = output[:, [1, 0, 3, 2]]
        np.save(image_path.with_suffix("_lines.npy"), lines)

def demo_region(image_path: str) -> None:
    """Process single image and plot region"""
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    region = np.array([])
    lsd_ext.lsd_scale_region(gray, 1, region)
    region = np.clip(region.T, 0, 255)
    cv.imwrite(Path(image_path).stem + "_region.png", region)

def demo(image_path: str) -> None:
    """Process single image and plot lines"""
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    output = lsd_ext.lsd(gray)
    lines = output[:, [1, 0, 3, 2]]
    plot_image = draw_lines(image, lines)
    cv.imwrite(Path(image_path).stem + ".png", plot_image)


if __name__ == "__main__":
    demo("lsd_1.6/chairs.pgm")
    demo_region("lsd_1.6/chairs.pgm")
