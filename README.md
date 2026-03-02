# LLM-based Fundamental and Price-Volume Factor Mining
*Status: Work in Progress*

# LLM Factor Automated Mining System - User Manual

## Quick Start

### 1. Environment Preparation

#### Install Dependencies
```bash
cd llm_factor_system
pip install -r requirements.txt
```

#### Configure API Keys (Optional)
The system prioritizes reading API configurations from environment variables. You can also configure them directly in the code:

**Windows:**
```bash
set CLOSEAI_API_KEY=your-api-key
set CLOSEAI_BASE_URL=https://api.openai-proxy.org/v1
```

**Linux/Mac:**
```bash
export CLOSEAI_API_KEY=your-api-key
export CLOSEAI_BASE_URL=https://api.openai-proxy.org/v1
```

Alternatively, configure them directly in `config.py`.

### 2. Data Preparation

Ensure your data files are organized in the following format (or modify the data loading logic to match your structure):

- **Minute-level Data**: `D:/Quant/data/minute_data/YYYY-MM-DD.parquet`
- **Daily-level Data**: `D:/Quant/data/daily_data/{field}.parquet`
- **Barra Factors**: `D:/Quant/data/daily_data/{factor_name}.parquet`

For detailed data format specifications, please refer to `Data_Format_Specifications.md`.

### 3. Running the System

#### Option 1: Direct Execution (Recommended)

**Batch Mining Mode** (Stops after mining a specific number of factors):
```bash
python run.py --mode batch --target-count 100
```

**Continuous Mining Mode** (Runs 24/7):
```bash
python run.py --mode continuous
```

**Using a Custom Configuration File**:
```bash
python run.py --mode batch --target-count 50 --config config.yaml
```

#### Option 2: Run via Python Script

Create a script named `start_mining.py`:
```python
from run import AlphaMiningSystem

# Initialize the system
system = AlphaMiningSystem()

# Batch Mining Mode
system.run_batch_mining(target_count=100)

# Or Continuous Mining Mode
# system.run_continuous_mining()
```

Then run:
```bash
python start_mining.py
```

#### Option 3: Interactive Execution

```python
from run import AlphaMiningSystem

# Initialize system
system = AlphaMiningSystem()

# Run batch mining
system.run_batch_mining(target_count=50)

# Check statistics
stats = system.slot_manager.get_stats()
print(stats)

# Stop the system
system.stop()
```

## Parameter Descriptions

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | str | `batch` | Running mode: `batch` or `continuous` |
| `--target-count` | int | 100 | Target number of factors for batch mode |
| `--config` | str | None | Path to custom configuration file |

### Configuration Parameters (config.py)

Key configuration items:

```python
{
    "data_paths": {
        "minute": "D:/Quant/data/minute_data",
        "daily": "D:/Quant/data/daily_data",
        "barra": "D:/Quant/data/daily_data"
    },
    "time_ranges": {
        "train_start": "2010-01-01",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2025-04-30"
    },
    "mining": {
        "max_parallel_slots": 8,        # Number of parallel worker slots
        "ic_threshold": 0.02,            # IC threshold for filtering
        "excess_return_threshold": 0.05,  # Excess return threshold
        "mmr_lambda": 0.7,               # MMR parameter λ for diversity
        "retry_times": 3                  # Number of retries on failure
    },
    "llm": {
        "model": "gpt-4.1-mini",
        "temperature": 0.1,
        "max_tokens": 2000
    }
}
```

## Running Modes Detail

### Batch Mining Mode

**Features**:
- Automatically stops after mining the specified number of factors.
- Ideal for rapid testing and validation.
- Generates a comprehensive final report.

**Use Cases**:
- Initial system testing.
- Validating system configuration.
- Quickly generating a small batch of factors.

**Example**:
```bash
python run.py --mode batch --target-count 50
```

### Continuous Mining Mode

**Features**:
- Runs 24/7 without interruption.
- Automatically saves progress periodically.
- Regularly updates RAG (Retrieval-Augmented Generation) indices.
- Performs daily automated factor synthesis.

**Use Cases**:
- Production environments.
- Long-term factor discovery.
- Continuous optimization of the factor library.

**Example**:
```bash
python run.py --mode continuous
```

**How to Stop**:
- Press `Ctrl+C` to send an interrupt signal.
- The system will automatically save progress and generate a final report before exiting.

## Output Files Description

The system generates the following files during operation:

### Log Files
- **Location**: `logs/alpha_mining_YYYYMMDD.log`
- **Content**: Execution logs, error messages, and debugging information.

### Report Files
- **Location**: `reports/`
- **Files**:
  - `status_YYYYMMDD_HHMM.json` - System status report.
  - `synthesis_price_YYYYMMDD_HHMM.json` - Price-volume factor synthesis report.
  - `synthesis_fundamental_YYYYMMDD_HHMM.json` - Fundamental factor synthesis report.

### Progress Files
- **Location**: `progress/progress_backup.json`
- **Content**: Backups of the Factor Pool and Idea Pool.

## Monitoring and Debugging

### View Logs
```bash
# Windows PowerShell
Get-Content logs/alpha_mining_*.log -Tail 50 -Wait

# Linux/Mac
tail -f logs/alpha_mining_*.log
```

### Check Latest Reports
```python
import json
import glob

# Get the latest status report
reports = glob.glob("reports/status_*.json")
if reports:
    latest = max(reports)
    with open(latest, 'r') as f:
        report = json.load(f)
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

### Inspect Factor Pool
```python
from factor.factor_pool import FactorPool

pool = FactorPool()
pool.load('progress/progress_backup.json')  # If a backup exists

factors = pool.get_all_factors()
print(f"Total Factors: {len(factors)}")
print(f"Average IC: {sum(f.ic_mean for f in factors) / len(factors):.4f}")
```

## FAQ

### 1. API Call Failure
**Problem**: `API Call Failed: ...`
**Solution**:
- Check if your API key is correct.
- Verify your internet connection.
- Check if the API Base URL is correct.
- Consult the log files for specific error codes.

### 2. Data File Not Found
**Problem**: `FileNotFoundError: Daily data file does not exist`
**Solution**:
- Verify data paths in `config.py`.
- Confirm the physical existence of the data files.
- Ensure the file formats match the system requirements.

### 3. Out of Memory
**Problem**: High memory usage after running for a long period.
**Solution**:
- Reduce the number of parallel slots (`max_parallel_slots`).
- Reduce the maximum capacity of the Factor Pool (`FactorPool(max_size=500)`).
- Periodically restart the system.

### 4. High Factor Generation Failure Rate
**Problem**: Most generated factors fail to execute.
**Solution**:
- Check if the LLM API is behaving normally.
- Lower the `temperature` parameter in `config.py` for higher stability.
- Check if the expression corrector is working.
- Review error messages in the logs to identify common syntax issues.

## Performance Optimization Tips

1. **Parallel Slots**: Adjust based on your CPU cores; 1-2x the number of physical cores is usually recommended.
2. **Data Caching**: The system automatically caches loaded data to avoid redundant disk I/O.
3. **Regular Cleanup**: Delete old log files and reports periodically to free up disk space.
4. **Resource Monitoring**: Use system tools to monitor CPU, RAM, and Disk I/O usage.

## Next Steps

- Review `Framework_Summary.md` for system architecture.
- Review `Update_Log.md` for the latest improvements.
- Review `Data_Format_Specifications.md` for detailed data requirements.

## Technical Support

If you encounter issues:
1. Check log files for detailed error traces.
2. Verify configuration settings.
3. Ensure data formats are compliant.
4. Consult the documentation provided in the repository.
