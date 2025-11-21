# Week 8 Quarter 1 Checkpoint 

Rosey Gutierrez DSC 180A TA Checkpoint

## Experiment: Response Grading with Rubric & injection 

## Contents

- `Folder: Week 5 Checkpoint` - Folder with code from a previous experiment for the Week 5 checkpoint.
- `Folder: Week 8 Checkpoint` - Folder with code from a more recent experiment for the Week 8 checkpoint. 
- `experiment.py` - Python script executable. Requires OpenRouter API key and a local install of Ollama with gpt-oss and qwen.    
- `analysis.ipynb` - Blank output jupyter notebook to reproduce results from the experiment. Should be able to run this after running experiment.py and getting the .csv data from it. Week 8 Checkpoint folder has full experiment notebook and data. 

## 1. Install Dependencies

**Required packages:**
```bash
pip install httpx aiometer aiosqlite jmespath tenacity tqdm pandas py-mini-racer
```

**Optional packages** (for additional export formats):
```bash
pip install openpyxl    # For Excel export
pip install pyarrow     # For Parquet export
```

### 2. Set API Keys (Required)

API keys **must** be set as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
# Add any other providers your experiment uses
```

The script checks all required keys upfront and will exit if any are missing.

### 3. Run the Experiment

**Basic usage:**
```bash
python experiment.py
```

### 4. Run Analysis

Rename .csv file to just `results.csv` for the notebook to work properly.
