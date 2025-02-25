import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_paper_area(img):
    """Detect the white paper area of the receipt"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur slightly to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Invert the image to make paper area white
    thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the paper contour
    paper_contour = None
    for contour in contours:
        # Get perimeter
        perimeter = cv2.arcLength(contour, True)
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it's roughly rectangular (4 corners) and has reasonable area
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            img_area = img.shape[0] * img.shape[1]
            # Check if area is reasonable (between 10% and 90% of image)
            if 0.1 * img_area < area < 0.9 * img_area:
                paper_contour = approx
                break
    
    if paper_contour is None:
        raise ValueError("Could not find receipt paper area")
    
    return paper_contour

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    # Convert points to numpy array
    pts = pts.reshape(4, 2)
    
    # Initialize ordered points
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left will have smallest sum
    # Bottom-right will have largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have smallest difference
    # Bottom-left will have largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def transform_receipt(img, corners):
    """Transform receipt to get straight-on view"""
    # Get ordered points
    rect = order_points(corners)
    
    # Calculate width
    width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    width = max(int(width_a), int(width_b))
    
    # Calculate height
    height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    height = max(int(height_a), int(height_b))
    
    # Create destination points
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Get perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped

def enhance_text(img):
    """Enhance text clarity"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Binary threshold
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def process_receipt(image_path):
    """Main function to process receipt"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Make copy for visualization
    img_with_corners = img.copy()
    
    # Detect paper area
    corners = detect_paper_area(img)
    
    # Draw detected corners
    cv2.drawContours(img_with_corners, [corners], -1, (0, 255, 0), 2)
    
    # Transform to straight-on view
    straightened = transform_receipt(img, corners)
    
    # Enhance text
    enhanced = enhance_text(straightened)
    
    return img_with_corners, straightened, enhanced

def display_results(detected, straightened, enhanced):
    """Display processing results"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    plt.title("Detected Paper Area")
    plt.axis("off")
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(straightened, cv2.COLOR_BGR2RGB))
    plt.title("Straightened")
    plt.axis("off")
    
    plt.subplot(133)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Text")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        image_path = "Bill2.jpg"
        detected, straightened, enhanced = process_receipt(image_path)
        display_results(detected, straightened, enhanced)
    except Exception as e:
        print(f"Error: {str(e)}")