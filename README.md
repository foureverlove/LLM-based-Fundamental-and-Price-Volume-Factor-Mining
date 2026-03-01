# LLM-based-Fundamental-and-Price-Volume-Factor-Mining
# LLM因子自动挖掘系统 - 使用说明

## 快速开始

### 1. 环境准备

#### 安装依赖
```bash
cd llm_factor_system
pip install -r requirements.txt
```

#### 配置API密钥（可选）
系统会优先从环境变量读取API配置，也可以在代码中配置：

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

或者在 `config.py` 中直接配置（不推荐，安全性较低）。

### 2. 数据准备

确保数据文件已按照以下格式准备好(或者自己改一下)：

- **分钟频数据**: `D:/量化/data/minute_data/YYYY-MM-DD.parquet`
- **日频数据**: `D:/量化/data/daily_data/{field}.parquet`
- **Barra因子**: `D:/量化/data/daily_data/{factor_name}.parquet`

详细的数据格式说明请参考 `数据格式说明.md`。

### 3. 运行系统

#### 方式一：直接运行（推荐）

**批量挖掘模式**（挖掘指定数量的因子后停止）：
```bash
python run.py --mode batch --target-count 100
```

**持续挖掘模式**（7×24小时运行）：
```bash
python run.py --mode continuous
```

**使用自定义配置文件**：
```bash
python run.py --mode batch --target-count 50 --config config.yaml
```

#### 方式二：Python脚本运行

创建脚本 `start_mining.py`：
```python
from run import AlphaMiningSystem

# 创建系统实例
system = AlphaMiningSystem()

# 批量挖掘模式
system.run_batch_mining(target_count=100)

# 或持续挖掘模式
# system.run_continuous_mining()
```

然后运行：
```bash
python start_mining.py
```

#### 方式三：交互式运行

```python
from run import AlphaMiningSystem

# 创建系统
system = AlphaMiningSystem()

# 批量挖掘
system.run_batch_mining(target_count=50)

# 查看统计信息
stats = system.slot_manager.get_stats()
print(stats)

# 停止系统
system.stop()
```

## 运行参数说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `batch` | 运行模式：`batch`（批量）或 `continuous`（持续） |
| `--target-count` | int | 100 | 批量模式下目标因子数量 |
| `--config` | str | None | 自定义配置文件路径 |

### 配置参数（config.py）

主要配置项：

```python
{
    "data_paths": {
        "minute": "D:/量化/data/minute_data",
        "daily": "D:/量化/data/daily_data",
        "barra": "D:/量化/data/daily_data"
    },
    "time_ranges": {
        "train_start": "2010-01-01",
        "train_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2025-04-30"
    },
    "mining": {
        "max_parallel_slots": 8,        # 并行槽位数
        "ic_threshold": 0.02,            # IC阈值
        "excess_return_threshold": 0.05,  # 超额收益阈值
        "mmr_lambda": 0.7,               # MMR参数λ
        "retry_times": 3                  # 重试次数
    },
    "llm": {
        "model": "gpt-4.1-mini",
        "temperature": 0.1,
        "max_tokens": 2000
    }
}
```

## 运行模式详解

### 批量挖掘模式

**特点**：
- 挖掘指定数量的因子后自动停止
- 适合快速测试和验证
- 会生成完整的报告

**使用场景**：
- 初次运行测试
- 验证系统配置
- 快速生成少量因子

**示例**：
```bash
python run.py --mode batch --target-count 50
```

### 持续挖掘模式

**特点**：
- 7×24小时不间断运行
- 自动定期保存进度
- 定期更新RAG索引
- 每日自动合成因子

**使用场景**：
- 生产环境
- 长期因子挖掘
- 持续优化因子库

**示例**：
```bash
python run.py --mode continuous
```

**停止方式**：
- 按 `Ctrl+C` 发送中断信号
- 系统会自动保存进度并生成报告

## 输出文件说明

系统运行后会生成以下文件：

### 日志文件
- **位置**: `logs/alpha_mining_YYYYMMDD.log`
- **内容**: 系统运行日志、错误信息、调试信息

### 报告文件
- **位置**: `reports/`
- **文件**:
  - `status_YYYYMMDD_HHMM.json` - 系统状态报告
  - `synthesis_price_YYYYMMDD_HHMM.json` - 量价因子合成报告
  - `synthesis_fundamental_YYYYMMDD_HHMM.json` - 基本面因子合成报告

### 进度文件
- **位置**: `progress/progress_backup.json`
- **内容**: 因子池和想法池的备份

## 监控和调试

### 查看日志
```bash
# Windows PowerShell
Get-Content logs/alpha_mining_*.log -Tail 50 -Wait

# Linux/Mac
tail -f logs/alpha_mining_*.log
```

### 查看最新报告
```python
import json
from datetime import datetime
import glob

# 获取最新状态报告
reports = glob.glob("reports/status_*.json")
if reports:
    latest = max(reports)
    with open(latest, 'r') as f:
        report = json.load(f)
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

### 查看因子池
```python
from factor.factor_pool import FactorPool

pool = FactorPool()
pool.load('progress/progress_backup.json')  # 如果保存了

factors = pool.get_all_factors()
print(f"因子总数: {len(factors)}")
print(f"平均IC: {sum(f.ic_mean for f in factors) / len(factors):.4f}")
```

## 常见问题

### 1. API调用失败
**问题**: `API调用失败: ...`

**解决方案**:
- 检查API密钥是否正确
- 检查网络连接
- 检查API Base URL是否正确
- 查看日志文件获取详细错误信息

### 2. 数据文件不存在
**问题**: `FileNotFoundError: 日频数据文件不存在`

**解决方案**:
- 检查数据路径是否正确（`config.py`）
- 确认数据文件是否存在
- 检查文件格式是否正确

### 3. 内存不足
**问题**: 运行一段时间后内存占用过高

**解决方案**:
- 减少并行槽位数（`max_parallel_slots`）
- 减少因子池最大容量（`FactorPool(max_size=500)`）
- 定期重启系统

### 4. 因子生成失败率高
**问题**: 生成的因子大部分无法执行

**解决方案**:
- 检查LLM API是否正常
- 调整temperature参数（降低以增加稳定性）
- 检查表达式修正器是否正常工作
- 查看日志中的错误信息

## 性能优化建议

1. **并行槽位数量**: 根据CPU核心数调整，建议为CPU核心数的1-2倍
2. **数据缓存**: 系统会自动缓存已加载的数据，避免重复读取
3. **定期清理**: 定期清理日志文件和旧报告，释放磁盘空间
4. **监控资源**: 使用系统监控工具观察CPU、内存、磁盘使用情况

## 下一步

- 查看 `框架总结.md` 了解系统架构
- 查看 `完善总结.md` 了解最新更新
- 查看 `数据格式说明.md` 了解数据格式要求

## 技术支持

如遇到问题，请：
1. 查看日志文件获取详细错误信息
2. 检查配置文件是否正确
3. 确认数据文件格式是否符合要求
4. 查看相关文档
