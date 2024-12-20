# Visualize Project

## Directory Structure

```
visualize/
├── __init__.py
├── main.py          # Main script to process data and create visualizations
├── requirements.txt # Python dependencies
├── data/            # Directory for raw data files (optional)
├── scripts/         # Directory for utility scripts (optional)
├── plots/           # Directory for storing generated plots (optional)
```

---

## Prerequisites

- Python 3.6 or later
- Virtual environment with required dependencies installed

---

## Setup

1. **Set up the Virtual Environment**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Script**

   ```
   python main.py
   ```

2. **Generated Output**
   - The script reads data from `../output/results.csv`.
   - A bar plot (`average_execution_time.png`) is saved in the `../output` directory.

---
