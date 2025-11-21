# Reproducibility Bundle: French Revolution Grading with Rubric & injection JSON - Nov 9, 07:50 PM

Generated: 2025-11-10T04:31:53.973Z
Template: **production** (Production-grade with SQLite, retries, progress bars)

## Contents
- `experiment.py` - Python script to reproduce the experiment (production template)
- `trial_config.json` - Complete trial configuration
- `data/results.csv` - Full experimental results in CSV format
- `data/results.xlsx` - Full experimental results in Excel format
- `data/results.jsonl` - Full experimental results in JSONL format (one JSON object per line)


---

## Quick Start (Production Template)

### 1. Install Dependencies

**Required packages:**
```bash
pip install httpx aiometer aiosqlite jmespath tenacity tqdm pandas
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

**With concurrency and rate limiting:**
```bash
python experiment.py --concurrent 10 --rate-limit 5.0
```
This runs 10 API calls concurrently, with a maximum of 5 requests per second.

**With custom output format:**
```bash
python experiment.py --output excel --concurrent 5
python experiment.py --output parquet --concurrent 5
```

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output` / `-o` | `csv` | Output format: `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `--concurrent` / `-c` | `10` | Number of concurrent API requests |
| `--rate-limit` / `-r` | `5.0` | Maximum requests per second |
| `--timeout` / `-t` | `90` | Request timeout in seconds |
| `--output-file` / `-f` | Auto | Custom output filename |
| `--resume` | Off | Resume from existing database |
| `--db-file` | Auto | Database file to use/resume from |

**Examples:**

```bash
# Fast batch processing
python experiment.py --concurrent 20 --rate-limit 10.0

# Conservative (avoid rate limits)
python experiment.py --concurrent 3 --rate-limit 1.0

# Long-running API calls
python experiment.py --timeout 180

# Resume from previous run
python experiment.py --resume --db-file results_20250128_143022.db

# Export to specific format
python experiment.py --output excel --output-file my_results.xlsx
```

---

## Features

### 🔥 Production-Grade Reliability

**Professional Retry Logic:**
- Up to 10 automatic retry attempts with exponential backoff (1s → 30s)
- Smart handling of rate limits (429) and server errors (500/502/503/504)
- Immediate failure on auth errors (401/403) - no wasted retries
- Detailed logging of retry attempts

**SQLite Persistence:**
- All results saved to SQLite database in real-time
- Atomic transactions ensure no data loss
- Query results with standard SQL tools
- Database survives crashes and interruptions

### ⚡ High Performance

**Concurrent Execution:**
- Run multiple API calls simultaneously with `--concurrent`
- Built-in rate limiting with `--rate-limit` to avoid throttling
- Smart task scheduling with aiometer

**Real-Time Progress:**
- Live progress bar showing completion status
- Success/failure counters updated in real-time
- Last tested model displayed
- ETA based on current rate

```
API Calls: 45%|████████      | 450/1000 [03:25<04:11, ✓:445 ✗:5 last:GPT-4]
```

### 💾 Multiple Export Formats

Choose your preferred output format:
- **CSV** - Opens in Excel, compatible with all tools
- **TSV** - Tab-separated, better for fields with commas
- **JSON** - Pretty-printed JSON array
- **JSONL** - One JSON object per line (streaming-friendly)
- **Excel** - Native .xlsx with proper formatting
- **Parquet** - Columnar format for big data tools

The database is always saved, export format is just for convenience.

### 🔄 Resume from Interruptions

Press Ctrl+C at any time. To resume:

```bash
python experiment.py --resume --db-file results_20250128_143022.db
```

The script will:
- Skip all previously successful tests
- Retry any previously failed tests
- Continue from where you left off
- Export with the updated results

### 📊 Smart Output Schema

**Dynamic columns based on your configuration:**
- Each parameter gets its own column: `param_temperature`, `param_max_tokens`, etc.
- Each variable gets its own column
- Full request/response payloads saved as JSON
- Extracted answers in dedicated column

**Example CSV structure:**
```
config_index | model_display_name | param_temperature | param_max_tokens | variable1 | variable2 | extracted | success | error
```

### 🎯 API Key Validation

The script validates all API keys upfront:

```
🔑 Checking API keys...
✅ OPENAI_API_KEY: Found (enables 3 models)
✅ ANTHROPIC_API_KEY: Found (enables 2 models)
❌ Missing API keys for 1 providers:
   GOOGLE_API_KEY (required for 1 models)

🔧 Please set the missing API keys:
   export GOOGLE_API_KEY='your-key-here'
```

### 📈 Detailed Summary

After completion, see results grouped by configuration:

```
==================================================
📈 FINAL RESULTS
==================================================
[1] ✅ GPT-4: 100/100 (100%)
[2] ✅ Claude 3 Opus: 98/100 (98%)
[3] ⚠️  Claude 3 Sonnet [Config 1]: 95/100 (95%)
[4] ⚠️  Claude 3 Sonnet [Config 2]: 92/100 (92%)

📊 View results: sqlite3 results_20250128_143022.db
   Example: SELECT * FROM results WHERE success = 0;
```

---

## Working with SQLite Database

The database file (`results_YYYYMMDD_HHMMSS.db`) contains all results in a structured format.

**Query examples:**

```bash
# Open the database
sqlite3 results_20250128_143022.db

# View schema
.schema results

# See all failures
SELECT model_display_name, error, COUNT(*)
FROM results
WHERE success = 0
GROUP BY model_display_name, error;

# Check success rates by model
SELECT model_display_name,
       COUNT(*) as total,
       SUM(success) as successful,
       ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_rate
FROM results
GROUP BY model_display_name;

# Export specific columns to CSV
.mode csv
.output filtered_results.csv
SELECT model_display_name, prompt, extracted, success FROM results;
.quit
```

---

## Performance Tuning

### Optimal Concurrency

- **Fast APIs (OpenAI, Anthropic):** `--concurrent 10-20`
- **Slower APIs or strict rate limits:** `--concurrent 3-5`
- **Local models (Ollama):** `--concurrent 1-2`

### Rate Limiting

Calculate based on your API tier:
- **OpenAI Tier 1:** ~500 requests/minute → `--rate-limit 8.0`
- **OpenAI Tier 2:** ~5000 requests/minute → `--rate-limit 80.0`
- **Anthropic Standard:** ~50 requests/minute → `--rate-limit 0.8`

**Formula:** `rate-limit = (requests_per_minute / 60) * 0.9` (90% to be safe)

### Balancing Speed and Safety

```bash
# Maximum throughput (risk of rate limits)
python experiment.py --concurrent 20 --rate-limit 10.0

# Balanced (recommended)
python experiment.py --concurrent 10 --rate-limit 5.0

# Conservative (guaranteed no rate limits)
python experiment.py --concurrent 3 --rate-limit 1.0
```

---

## Troubleshooting

### Rate Limit Errors (429)
```
🔄 Retry attempt 3 for GPT-4 after HTTPStatusError: 429 Rate Limit
```
**Fix:** Reduce `--concurrent` or `--rate-limit`, or wait and use `--resume`

### Missing Dependencies
```
⚠️  Excel export requires 'openpyxl'. Install with: pip install openpyxl
   Falling back to CSV export...
```
**Info:** The script automatically falls back to CSV if optional dependencies are missing

### Extraction Failures
```
⚠️  GPT-4: Couldn't extract content. Tried: choices[0].message.content, data.content
```
**Fix:** Check the API response format in the database (`response` column) and update `extract_paths` in the script

### Network Timeouts
```
🌐 GPT-4: Network error - TimeoutException
```
**Fix:** Increase `--timeout` or check your network connection

---

## Tips for Large Experiments

1. **Test first:** Run with one model to verify everything works
2. **Use --resume:** For experiments with 1000+ calls, use resume to handle interruptions
3. **Monitor progress:** Watch the progress bar and success/fail counters
4. **Check database during run:** Use `sqlite3` to query partial results while script is running
5. **Export after completion:** Re-export to different formats without re-running:
   ```bash
   python experiment.py --resume --db-file results.db --output excel
   ```
6. **Adjust based on errors:** If you see many failures, Ctrl+C and adjust settings before resuming

---

## Cost Management

- Each API call costs money - check your provider's pricing
- Use `--concurrent` and `--rate-limit` to control spending rate
- Monitor the progress bar to estimate total cost during execution
- Consider testing with a small subset first
- Some providers offer usage caps - set them before running large experiments

