# 💎 DiamondVision Elite: Enterprise Grading OS

DiamondVision Elite is an end-to-end B2B Machine Learning application designed specifically for the Surat diamond wholesale market (Varachha/Mahidharpura). It combines Computer Vision and regression modeling to automate diamond lot sorting and generate live wholesale valuations.

## 📸 System Interface
<img width="1882" height="502" alt="image" src="https://github.com/user-attachments/assets/7e55d91d-692c-4399-b183-6d3d186edb11" />
<img width="664" height="713" alt="image" src="https://github.com/user-attachments/assets/16cb2a08-1389-4e1f-8cee-a29eb271cac3" />
<img width="465" height="341" alt="image" src="https://github.com/user-attachments/assets/7f0fc001-ba4b-4706-92e0-9cc9e969f1e0" />
<img width="874" height="701" alt="image" src="https://github.com/user-attachments/assets/015f50cf-b5ed-4ed2-b35d-3618b27a21e1" />

<img width="235" height="736" alt="image" src="https://github.com/user-attachments/assets/39380d7e-32e2-4d42-84b7-ac4e1edd2048" />


<img width="1663" height="656" alt="image" src="https://github.com/user-attachments/assets/97f49fff-400d-4abb-bd1d-66a7f8e7c45a" />
<img width="1507" height="386" alt="image" src="https://github.com/user-attachments/assets/d2456640-ab77-42d0-bc5a-93c1c8b77cc1" />
<img width="1656" height="727" alt="image" src="https://github.com/user-attachments/assets/23c756b3-3a2b-4eb4-9004-4aa4968b9121" />
<img width="1635" height="498" alt="image" src="https://github.com/user-attachments/assets/fca87652-e00f-4cb5-8d69-771ce8808f55" />





## 🚀 Key Features

* **AI Optical Geometry Analysis:** Utilizes a custom-trained **MobileNetV2** Convolutional Neural Network (CNN) to classify diamond cuts with high precision.
* **Dynamic Edge-Case Thresholds:** Features strict dynamic confidence thresholds (95%) specifically engineered to trap and flag complex optical edge-case geometries like Marquise and Cushion cuts for manual verification. 
* **Live XGBoost Pricing Engine:** Computes wholesale estimates in INR via a deployed XGBoost regression model, factoring in Carat, Color, Clarity, Cut, and standard manufacturing hub discounts.
* **Bulk Lot BI Dashboard:** Processes multiple specimens simultaneously, generating automated Business Intelligence (BI) insights, composition pie charts (via Plotly), and an interactive data manifest.
* **Human-in-the-Loop Architecture:** Features an editable data grid (`st.data_editor`) that flags low-confidence AI predictions for manual grader review, ensuring 100% enterprise data integrity before export.

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Custom CSS (Glassmorphism UI)
* **Computer Vision:** TensorFlow, Keras (MobileNetV2 Transfer Learning), Pillow, OpenCV
* **Data Processing & Analytics:** Pandas, NumPy
* **Data Visualization:** Plotly Express
* **Machine Learning (Tabular):** Scikit-Learn, XGBoost

## 💻 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/prakritim01/diamondvision-elite.git](https://github.com/prakritim01/diamondvision-elite.git)
cd diamondvision-elite
2. Install Dependencies
pip install -r requirements.txt
3. Launch the Streamlit OS
streamlit run app.py

📁 Project Structure
DiamondVision_Elite/
├── data/                       # Raw and processed market data (ignored in Git)
├── models/                     # Saved model artifacts
│   ├── shape_model.h5          # CNN Vision Model
│   ├── price_model.pkl         # Pricing Regression Model
│   └── classes.json            # Class indices for shape mapping
├── notebooks/                  # Jupyter notebooks for model training & EDA
│   └── 02_image_training.ipynb # Data auditing and CNN training pipeline
├── src/
│   ├── data_prep.py            # Data cleaning scripts
│   ├── image_processor.py      # Vision augmentation scripts
│   └── pricing_logic.py        # XGBoost training pipeline
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
