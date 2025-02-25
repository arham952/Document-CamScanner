# 📄 Document-CamScanner 

**Document CamScanner** is an AI-powered **receipt/document scanner** that enhances scanned text for better readability.  
It uses **Computer Vision (CV) and Image Processing techniques** to detect, transform, and enhance documents from images.

---

## 🚀 Features  
✅ **Automatic paper/receipt detection** using contours  
✅ **Perspective transformation** for a straight-on document view  
✅ **Text enhancement** for better readability  
✅ **Noise reduction and contrast improvement**  
✅ **Matplotlib visualization** of processing results  

---

## **🛠 Technologies Used**
| Technology        | Purpose |
|------------------|---------|
| **Python**       | Backend scripting |
| **OpenCV**       | Image processing (contours, transformations) |
| **NumPy**        | Array operations |
| **Matplotlib**   | Displaying image processing results |

---

## **📥 Installation & Usage**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/arham952/Document-CamScanner.git
cd Document-CamScanner

📊 How It Works
1️⃣ Detects paper area using contours and adaptive thresholding
2️⃣ Applies perspective transformation for a straight-on document view
3️⃣ Enhances text clarity using CLAHE (contrast enhancement)
4️⃣ Removes noise while preserving text edges
5️⃣ Displays results: Detected, Straightened & Enhanced document

🎯 Processing Steps & Example Output

🔹 Detected Paper Area
🔹 Transformed to a Straight View
🔹 Enhanced for Clearer Text
💡 The processed images are displayed using Matplotlib.

⚠ Troubleshooting
If you get "Could not read image", check if Bill2.jpg is in the correct folder.
If OpenCV functions fail, reinstall dependencies using:

pip install --upgrade opencv-python

📩 Contact
👤 Muhammad Arham
📧 Email: m.arham1264@gmail.com
🔗 LinkedIn: linkedin.com/in/muhammad-arham-95b8331a4
