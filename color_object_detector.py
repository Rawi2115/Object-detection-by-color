import cv2
import numpy as np
import os

def hex_to_bgr(hex_color: str):
    """Convert hex color to BGR format."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format. Use format like #FF5733")
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        raise ValueError("Invalid hex color format. Use format like #FF5733")
    return (b, g, r)

def hex_to_hsv_range(hex_color: str, tolerance=15):
    """Convert hex color to HSV range with given tolerance."""
    bgr = np.array([[hex_to_bgr(hex_color)]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    
    # Use dynamic S/V ranges based on input color
    lower_s = max(s - 100, 0)
    lower_v = max(v - 100, 0)
    
    # Handle red color wrapping (hue is circular: 0-179)
    if h < tolerance:
        # Red wraps from high values to low values
        lower1 = np.array([0, lower_s, lower_v], dtype=np.uint8)
        upper1 = np.array([h + tolerance, 255, 255], dtype=np.uint8)
        lower2 = np.array([180 - (tolerance - h), lower_s, lower_v], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        return (lower1, upper1, lower2, upper2)
    elif h > 179 - tolerance:
        # Red wraps from low values to high values
        lower1 = np.array([h - tolerance, lower_s, lower_v], dtype=np.uint8)
        upper1 = np.array([179, 255, 255], dtype=np.uint8)
        lower2 = np.array([0, lower_s, lower_v], dtype=np.uint8)
        upper2 = np.array([tolerance - (179 - h), 255, 255], dtype=np.uint8)
        return (lower1, upper1, lower2, upper2)
    else:
        # Normal case (no wrapping)
        lower = np.array([h - tolerance, lower_s, lower_v], dtype=np.uint8)
        upper = np.array([h + tolerance, 255, 255], dtype=np.uint8)
        return (lower, upper)


def detect_by_color(image_path: str, hex_color: str, object_name: str="object"):
    """Detect objects of a specific color in an image."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hsv_range = hex_to_hsv_range(hex_color)
    
    # Handle red color wrapping (two ranges)
    if len(hsv_range) == 4:
        lower1, upper1, lower2, upper2 = hsv_range
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = hsv_range
        mask = cv2.inRange(hsv_image, lower, upper)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = False
    for c in contours:
        area = cv2.contourArea(c)
        if area < 400:
            continue
        found = True
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, object_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if not found:
        print(f"No {object_name} detected.")
    
    # Save the output image with _detected suffix
    base_name = os.path.splitext(image_path)[0]
    extension = os.path.splitext(image_path)[1]
    output_path = f"{base_name}_detected{extension}"
    cv2.imwrite(output_path, image)
    print(f"Detected image saved as: {output_path}")
    
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("=== Color-Based Object Detector ===")
    image_path = input("Enter the path to the image file: ")
    hex_color = input("Enter the hex color code of the object to detect (e.g., #FF5733): ")
    object_name = input("Enter the name of the object (e.g., 'ball', 'box'): ")
    
    try:
        detect_by_color(image_path, hex_color, object_name)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()