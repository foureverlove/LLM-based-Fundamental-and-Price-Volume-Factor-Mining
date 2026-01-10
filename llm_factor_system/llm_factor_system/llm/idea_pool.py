# ==================== llm/idea_pool.py ====================
"""
改进想法池
"""
import threading
import random
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta

class IdeaPool:
    """改进想法池"""
    
    def __init__(self, max_size: int = 1000):
        self.ideas = []  # 存储改进想法
        self.idea_scores = {}  # 想法效果评分
        self.lock = threading.Lock()
        self.max_size = max_size
        
        # 初始化一些示例想法（根据研报中的案例）
        self._init_example_ideas()
    
    def _init_example_ideas(self):
        """初始化示例想法（根据研报）"""
        example_ideas = [
            {
                "idea": "趋势类因子应配对应周期的整指标（如5日趋势配5日整低），并叠加估值类因子控制泡沫风险",
                "category": "trend",
                "score": 0.8,
                "source": "研报案例",
                "applicable_operators": ["Slope", "EMA", "Mean", "Ref"]
            },
            {
                "idea": "用相关性过滤策略，用变化量替代波动指数和波动速信号",
                "category": "filter",
                "score": 0.7,
                "source": "研报案例",
                "applicable_operators": ["Corr", "Cov", "Delta", "Slope"]
            },
            {
                "idea": "分母采用多字段复合运算消除单维度偏差，提升信号锐度同时降低分母的波动干扰",
                "category": "normalization",
                "score": 0.75,
                "source": "研报案例",
                "applicable_operators": ["Std", "Var", "Mean", "Mul", "Div"]
            },
            {
                "idea": "引入对数处理稳定成交量信号，减少极端值影响",
                "category": "transformation",
                "score": 0.6,
                "source": "常规优化",
                "applicable_operators": ["Log", "Sign", "Abs"]
            },
            {
                "idea": "价格波动使用EMA(10日)加强近期敏感度",
                "category": "smoothing",
                "score": 0.65,
                "source": "研报案例",
                "applicable_operators": ["EMA", "Mean", "Std"]
            },
            {
                "idea": "波动率压缩时趋势更脆弱，用协方差30日比相关10日更能捕捉中期背离",
                "category": "timing",
                "score": 0.8,
                "source": "研报案例",
                "applicable_operators": ["Cov", "Corr", "Std"]
            }
        ]
        
        for idea in example_ideas:
            self.add_idea(idea["idea"], idea.get("category"), 
                         idea.get("score"), idea.get("source"))
    
    def add_idea(self, idea_text: str, category: str = None, 
                initial_score: float = 0.5, source: str = "auto_generated"):
        """添加改进想法"""
        with self.lock:
            # 避免重复
            if idea_text in [idea["text"] for idea in self.ideas]:
                return False
            
            idea_data = {
                "text": idea_text,
                "category": category or self._categorize_idea(idea_text),
                "score": initial_score,
                "source": source,
                "created_time": datetime.now().isoformat(),
                "usage_count": 0,
                "success_count": 0
            }
            
            self.ideas.append(idea_data)
            
            # 保持池大小
            if len(self.ideas) > self.max_size:
                # 移除评分最低的想法
                self.ideas.sort(key=lambda x: x["score"])
                self.ideas = self.ideas[:self.max_size]
            
            return True
    
    def _categorize_idea(self, idea_text: str) -> str:
        """自动分类想法"""
        categories = {
            "trend": ["趋势", "动量", "slope", "momentum", "方向"],
            "mean_reversion": ["反转", "均值回归", "reversion", "overbought", "oversold"],
            "volume": ["成交量", "volume", "量价", "资金流"],
            "volatility": ["波动率", "volatility", "风险", "std", "var"],
            "correlation": ["相关", "corr", "cov", "联动", "同步"],
            "normalization": ["标准化", "归一化", "zscore", "scale", "量纲"],
            "timing": ["时机", "timing", "窗口", "周期", "参数"]
        }
        
        idea_lower = idea_text.lower()
        for category, keywords in categories.items():
            if any(keyword in idea_lower for keyword in keywords):
                return category
        
        return "general"
    
    def get_random_ideas(self, n: int = 3, 
                        category: str = None,
                        min_score: float = 0.3) -> List[str]:
        """随机获取想法"""
        with self.lock:
            filtered = self.ideas
            
            if category:
                filtered = [idea for idea in filtered if idea["category"] == category]
            
            filtered = [idea for idea in filtered if idea["score"] >= min_score]
            
            if not filtered:
                return []
            
            # 按评分加权随机选择
            weights = [idea["score"] for idea in filtered]
            selected = random.choices(filtered, weights=weights, k=min(n, len(filtered)))
            
            # 增加使用计数
            for idea in selected:
                idea["usage_count"] += 1
            
            return [idea["text"] for idea in selected]
    
    def update_idea_score(self, idea_text: str, success: bool):
        """更新想法评分"""
        with self.lock:
            for idea in self.ideas:
                if idea["text"] == idea_text:
                    idea["usage_count"] += 1
                    if success:
                        idea["success_count"] += 1
                    
                    # 计算新的评分：成功率 + 时效性（新想法权重更高）
                    if idea["usage_count"] > 0:
                        success_rate = idea["success_count"] / idea["usage_count"]
                        
                        # 时效性衰减：创建时间越久，权重越低
                        created_time = datetime.fromisoformat(idea["created_time"])
                        days_old = (datetime.now() - created_time).days
                        recency_weight = max(0, 1 - days_old / 30)  # 30天衰减期
                        
                        idea["score"] = 0.7 * success_rate + 0.3 * recency_weight
                    break
    
    def get_high_score_ideas(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取高分想法"""
        with self.lock:
            sorted_ideas = sorted(self.ideas, key=lambda x: x["score"], reverse=True)
            return sorted_ideas[:n]
    
    def search_ideas(self, query: str, n: int = 5) -> List[str]:
        """搜索相关想法"""
        with self.lock:
            # 简单关键词匹配
            query_lower = query.lower()
            matched = []
            
            for idea in self.ideas:
                if query_lower in idea["text"].lower():
                    matched.append(idea)
            
            # 按评分排序
            matched.sort(key=lambda x: x["score"], reverse=True)
            return [idea["text"] for idea in matched[:n]]
    
    def save(self, path: str):
        """保存想法池"""
        with self.lock:
            with open(path, 'w') as f:
                json.dump(self.ideas, f, indent=2)
    
    def load(self, path: str):
        """加载想法池"""
        with self.lock:
            with open(path, 'r') as f:
                self.ideas = json.load(f)