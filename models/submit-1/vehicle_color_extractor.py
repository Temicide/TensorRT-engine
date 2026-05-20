from __future__ import annotations

import cv2
import numpy as np


STANDARD_COLORS = [
    "Black",
    "Blue",
    "Blue-White",
    "Bronze",
    "Bronze Gold",
    "Bronze Gray",
    "Bronze Silver",
    "Charcoal",
    "Chartreuse",
    "Dark Green",
    "Gold",
    "Gray",
    "Green",
    "Light Green",
    "Maroon",
    "Metallic Green",
    "Navy Blue",
    "Olive Green",
    "Orange",
    "Pink",
    "Red",
    "Red-White",
    "Silver",
    "Slate Blue",
    "White",
    "Yellow",
    "Yellow-Green",
]


def center_body_crop(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    x1, x2 = int(w * 0.12), int(w * 0.88)
    y1, y2 = int(h * 0.34), int(h * 0.84)
    body = crop_bgr[y1:y2, x1:x2]
    return body if body.size else crop_bgr


def hsv_name_from_bgr(bgr: np.ndarray) -> str:
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = [int(x) for x in hsv]

    if v < 45:
        return "Black"
    if s < 32:
        if v > 185:
            return "White"
        if v > 128:
            return "Silver"
        if v > 105:
            return "Gray"
        if v < 78:
            return "Black"
        return "Charcoal"

    if s < 60 and v < 95:
        return "Charcoal"

    if h <= 6 or h >= 172:
        return "Maroon" if v < 120 else "Red"
    if 7 <= h <= 16:
        if v < 120:
            return "Bronze"
        return "Orange"
    if 17 <= h <= 26:
        if s < 95 and v < 170:
            return "Bronze"
        if v > 175 and s > 95:
            return "Gold"
        return "Bronze Gold"
    if 27 <= h <= 36:
        return "Yellow" if v > 150 else "Gold"
    if 37 <= h <= 48:
        return "Chartreuse" if s > 95 else "Yellow-Green"
    if 49 <= h <= 68:
        if v > 175:
            return "Light Green"
        if s < 85:
            return "Metallic Green"
        return "Green"
    if 69 <= h <= 88:
        return "Olive Green" if v < 145 else "Dark Green"
    if 89 <= h <= 103:
        return "Slate Blue" if s < 90 else "Blue"
    if 104 <= h <= 128:
        if v < 110:
            return "Navy Blue"
        return "Blue"
    if 129 <= h <= 145:
        return "Slate Blue"
    if 146 <= h <= 171:
        return "Pink" if v > 145 else "Maroon"
    return "Gray"


def combine_colors(primary: str, secondary: str, secondary_ratio: float) -> str:
    if secondary_ratio < 0.18:
        return primary

    pair = {primary, secondary}
    if pair == {"Blue", "White"}:
        return "Blue-White"
    if pair == {"Red", "White"}:
        return "Red-White"
    if pair == {"Yellow", "Green"} or pair == {"Yellow", "Light Green"}:
        return "Yellow-Green"
    if primary == "Bronze" and secondary == "Gold":
        return "Bronze Gold"
    if primary == "Bronze" and secondary == "Gray":
        return "Bronze Gray"
    if primary == "Bronze" and secondary == "Silver":
        return "Bronze Silver"
    return primary


def achromatic_color(hsv_pixels: np.ndarray) -> tuple[str, float] | None:
    if len(hsv_pixels) == 0:
        return None

    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]
    low_sat = s < 52
    very_dark = v < 72
    dark = v < 105
    bright = v > 178
    low_sat_ratio = float(low_sat.mean())
    dark_ratio = float(very_dark.mean())
    soft_dark_ratio = float(dark.mean())
    bright_ratio = float(bright.mean())

    if low_sat_ratio < 0.48 and dark_ratio < 0.50:
        return None

    if (dark_ratio > 0.42 and low_sat_ratio > 0.48) or (
        soft_dark_ratio > 0.62 and bright_ratio < 0.18 and low_sat_ratio > 0.42
    ):
        return ("Black", max(dark_ratio, soft_dark_ratio))

    if low_sat_ratio < 0.55:
        return None

    values = v[low_sat]
    if len(values) == 0:
        values = v
    v25, v50, v75 = [float(x) for x in np.percentile(values, [25, 50, 75])]

    if bright_ratio > 0.34 and v75 > 192 and soft_dark_ratio < 0.35:
        return ("White", max(bright_ratio, low_sat_ratio))

    if v50 < 102 and v75 < 150:
        return ("Black", max(soft_dark_ratio, low_sat_ratio * 0.75))
    if v50 < 124 and v75 < 168:
        return ("Charcoal", max(soft_dark_ratio, low_sat_ratio * 0.7))

    if v50 > 188 or (v75 > 218 and v25 > 118):
        return ("White", low_sat_ratio)
    if v50 > 138 or v75 > 178:
        return ("Silver", low_sat_ratio)
    if v50 > 104:
        return ("Gray", low_sat_ratio)
    return ("Black", low_sat_ratio)


def filtered_pixels(body_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2HSV)
    pixels = body_bgr.reshape(-1, 3)
    hsv_pixels = hsv.reshape(-1, 3)

    # Remove windows, tire gaps, hard shadows, and under-car road as much as possible.
    non_shadow = hsv_pixels[:, 2] > 68
    if int(non_shadow.sum()) > 300:
        pixels = pixels[non_shadow]
        hsv_pixels = hsv_pixels[non_shadow]

    # Drop tiny specular highlights that can pull white/silver cars too bright.
    not_glare = ~((hsv_pixels[:, 1] < 25) & (hsv_pixels[:, 2] > 245))
    if int(not_glare.sum()) > 300:
        pixels = pixels[not_glare]

    return pixels


def choose_cluster_color(centers: np.ndarray, counts: np.ndarray) -> tuple[str, float]:
    order = counts.argsort()[::-1]
    names = [hsv_name_from_bgr(centers[int(idx)]) for idx in order]
    ratios = [float(counts[int(idx)] / max(counts.sum(), 1)) for idx in order]

    primary = names[0]
    primary_ratio = ratios[0]
    secondary = names[1] if len(names) > 1 else primary
    secondary_ratio = ratios[1] if len(ratios) > 1 else 0.0

    # Dark blue/black clusters are often windshield or shadow. Prefer the body-colored
    # secondary cluster when it occupies enough area.
    shadow_like = {"Black", "Charcoal", "Navy Blue", "Slate Blue"}
    body_like = {"White", "Silver", "Gray", "Red", "Orange", "Yellow", "Gold", "Green", "Blue"}
    if primary in shadow_like and secondary in body_like and secondary_ratio >= 0.23 and primary_ratio < 0.68:
        return secondary, secondary_ratio

    return combine_colors(primary, secondary, secondary_ratio), primary_ratio


def dominant_color_kmeans(crop_bgr: np.ndarray, k: int = 3) -> tuple[str, float]:
    body = center_body_crop(crop_bgr)
    body = cv2.resize(body, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv.reshape(-1, 3)

    achromatic = achromatic_color(hsv_pixels)
    if achromatic is not None:
        return achromatic

    filtered = filtered_pixels(body)
    if len(filtered) < 200:
        filtered = body.reshape(-1, 3)

    data = np.float32(filtered)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    color, confidence = choose_cluster_color(centers, counts)
    return color, round(confidence, 4)


def dominant_color_hsv(crop_bgr: np.ndarray) -> tuple[str, float]:
    body = center_body_crop(crop_bgr)
    body = cv2.resize(body, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv.reshape(-1, 3)

    achromatic = achromatic_color(hsv_pixels)
    if achromatic is not None:
        return achromatic

    mask = hsv[:, :, 2] > 68
    pixels = body[mask]
    if len(pixels) < 200:
        pixels = body.reshape(-1, 3)

    median_bgr = np.median(pixels, axis=0)
    color = hsv_name_from_bgr(median_bgr)
    return color, 1.0


def estimate_vehicle_color(crop_bgr: np.ndarray, method: str = "kmeans") -> tuple[str, float]:
    if method == "hsv":
        return dominant_color_hsv(crop_bgr)
    if method == "kmeans":
        return dominant_color_kmeans(crop_bgr)
    raise ValueError(f"Unknown color method: {method}")
