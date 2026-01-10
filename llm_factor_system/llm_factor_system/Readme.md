# LLM因子挖掘框架

基于国金证券研报《基于LLM的全天候财务逻辑因子挖掘框架》实现的自动化因子挖掘系统。

## 核心特性

1. **7×24小时自动化运行**：多槽位并行挖掘，持续优化因子库
2. **改进的MMR选择机制**：同时考虑截面相关性和时序相关性，控制Barra风险暴露
3. **RAG启发机制**：借鉴成熟因子模式，平衡创造性和实用性
4. **自适应反馈机制**：idea池持续学习改进策略
5. **严格的样本外验证**：训练期(2010-2019)，测试期(2020-2025.04)

## 项目结构
llm_factor_mining/
├── config.py # 配置管理
├── run.py # 主运行脚本
├── README.md
│
├── data/ # 数据加载模块
│ ├── init.py
│ ├── minute_loader.py
│ ├── daily_loader.py
│ └── barra_loader.py
│
├── factor/ # 因子相关模块
│ ├── init.py
│ ├── factor.py # 因子类定义
│ ├── factor_pool.py # 因子池管理
│ ├── selector.py # MMR因子选择器
│ ├── evaluator.py # 因子评估器
│ └── synthesizer.py # 因子合成器
│
├── llm/ # LLM相关模块
│ ├── init.py
│ ├── generator.py # 因子生成器
│ ├── rag_handler.py # RAG处理器
│ └── idea_pool.py # 改进想法池
│
├── dsl/ # 领域特定语言
│ ├── init.py
│ ├── parser.py # 表达式解析器
│ ├── executor.py # 表达式执行器
│ └── expression_corrector.py # 表达式修正器
│
├── operators/ # 算子库
│ ├── init.py
│ ├── price_operators.py # 量价算子
│ └── fundamental_operators.py # 基本面算子
│
├── mining/ # 挖掘引擎
│ ├── init.py
│ ├── mining_engine.py # 挖掘引擎
│ └── slot_manager.py # 槽位管理器
│
├── logs/ # 日志目录
├── reports/ # 报告输出目录
└── progress/ # 进度保存目录

### 1. 安装依赖
```bash
pip install pandas numpy openai sentence-transformers faiss-cpu scikit-learn schedule
export OPENAI_API_KEY="your-api-key"
# 批量挖掘模式
python run.py --mode batch --target-count 100

# 持续挖掘模式
python run.py --mode continuous
tail -f logs/alpha_mining_*.log

# 查看最新报告
ls -la reports/