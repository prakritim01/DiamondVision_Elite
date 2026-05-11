# 💎 DiamondVision Elite: Enterprise Grading OS

DiamondVision Elite is an end-to-end B2B Machine Learning application designed specifically for the Surat diamond wholesale market (Varachha/Mahidharpura). It combines Computer Vision and regression modeling to automate diamond lot sorting and generate live wholesale valuations.

## 🚀 Key Features

* **AI Optical Geometry Analysis:** Utilizes a custom-trained **MobileNetV2** Convolutional Neural Network (CNN) to classify diamond cuts with high precision, explicitly tuned with data augmentation to handle complex edge-case geometries like Marquise and Cushion cuts.
* **Bulk Lot BI Dashboard:** Processes multiple specimens simultaneously, generating automated Business Intelligence (BI) insights, composition pie charts (via Plotly), and an interactive data manifest.
* **Human-in-the-Loop Architecture:** Features an editable data grid (`st.data_editor`) that flags low-confidence AI predictions ( < 90%) for manual grader review, ensuring 100% enterprise data integrity.
* **Surat Pricing Engine:** Computes wholesale estimates in INR based on real-world market parameters (Carat, Color, Clarity, Cut) factoring in standard manufacturing hub discounts.

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Custom CSS (Glassmorphism UI)
* **Computer Vision:** TensorFlow, Keras (MobileNetV2 Transfer Learning), Pillow, OpenCV
* **Data Processing & Analytics:** Pandas, NumPy
* **Data Visualization:** Plotly Express
* **Machine Learning (Tabular):** Scikit-Learn, XGBoost

## 📁 Project Structure

```text
DiamondVision_Elite/
├── data/                       # Raw and processed market data (ignored in Git)
├── models/                     # Saved model artifacts
│   ├── shape_model.h5          # CNN Vision Model
│   ├── price_model.pkl         # Pricing Regression Model
│   └── classes.json            # Class indices for shape mapping
├── notebooks/                  # Jupyter notebooks for model training & EDA
│   └── 02_image_training.ipynb # Data auditing and CNN training pipeline
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

