# Classification of the degree of diabetic retinopathy

The purpose of the project is to create a neural network to properly classify the severity of diabetic retinopathy on a scale of 0 to 4:

* 0 - No DR  
* 1 - Mild  
* 2 - Moderate  
* 3 - Severe  
* 4 - Proliferative DR

---

## What is Diabetic Retinopathy?

Diabetic Retinopathy (DR) is a **complication of diabetes** that affects the **blood vessels in the retina**, the light-sensitive tissue at the back of the eye. High blood sugar levels over time can damage these tiny blood vessels, leading to leakage, swelling, or blockage. Left untreated, DR can eventually cause **vision impairment and blindness**. Early detection and classification are crucial for timely clinical intervention. 

---

## Severity Scale Used in APTOS 2019 Blindness Detection

In the **APTOS 2019 Blindness Detection** dataset (hosted on Kaggle), each retinal image was **clinically rated by an expert** according to the severity of diabetic retinopathy. 

### DR Severity Levels

| Label | Stage | Description |
|-------|-------|-------------|
| **0** | **No DR** | No visible signs of diabetic retinopathy - retina appears healthy. |
| **1** | **Mild** | Early signs, such as small microaneurysms (tiny bulges in retinal blood vessels). |
| **2** | **Moderate** | More significant changes - more blood vessels are affected and may begin to close off. |
| **3** | **Severe** | Widespread blood vessel blockages and extensive retinal damage. |
| **4** | **Proliferative DR** | Most advanced stage - abnormal new blood vessel growth and bleeding; high risk of vision loss. |

---

## About the Dataset

The dataset consists of thousands of **fundus (retinal) images** taken under various imaging conditions and labeled with the DR severity level by clinicians. 

---

## Installation (Windows)


| Step | Windows (PowerShell) | Linux / macOS (bash) |
|-----:|----------------------|----------------------|
| **1. Clone repository** | `git clone https://github.com/Luki10011/PAOM-retinapatia.git` | `git clone https://github.com/Luki10011/PAOM-retinapatia.git` |
| **2. Enter project directory** | `cd PAOM-retinapatia` | `cd PAOM-retinapatia` |
| **3. Create virtual environment** | `python -m venv .venv` | `python3 -m venv .venv` |
| **4. Activate virtual environment** | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| **5. Install dependencies** | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

## Project structure

```text
PAOM_retinopathy/
├── data/
│   ├── processed/
│   ├── raw/
├── notebooks/
├── src/
│   ├── models/
│   ├── datamodules/
│   ├── training/
│   ├── utils/
│   └── main.py 
├── .venv/
├── README.md
└── requirements.txt
```

## Getting started

In order to download the dataset, you need to create an account at Kaggle. This project is based on **APTOS 2019 Blindness Detection** competition, which you can find under this link:
> https://www.kaggle.com/competitions/aptos2019-blindness-detection/submissions

After creating an account and accepting the rules of copetition, remember to set your environment variable **KAGGLE_API_TOKEN** to the value of your personal API Key:
* Windows (PowerShell)
    ```
    $env:KAGGLE_API_TOKEN = "<YOUR_API_TOKEN>"
    ```
* Linux
    ```
    export KAGGLE_API_TOKEN = <YOUR_API_TOKEN>
    ```
In order to check if everything is set up correctly run this command:
```
kaggle competitions list
```

To download the datset use the command below:
```
python .\src\utils\data_preprocessor.py
```