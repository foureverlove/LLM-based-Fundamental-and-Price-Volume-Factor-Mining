# ==================== llm/rag_handler.py ====================
"""
RAG处理器
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import pickle
import json
import os

# #region agent log
log_path = r"c:\Users\qtx27\Desktop\llm_factor_system\.cursor\debug.log"
try:
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"import-check","hypothesisId":"H1,H2,H3,H4","location":"rag_handler.py:9","message":"尝试导入faiss","data":{"step":"before_import"},"timestamp":int(__import__('time').time()*1000)}) + "\n")
except: pass
# #endregion

# 尝试导入faiss，如果失败则使用替代方案
try:
    import faiss
    FAISS_AVAILABLE = True
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"import-check","hypothesisId":"H1","location":"rag_handler.py:15","message":"faiss导入成功","data":{"available":True},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
except ImportError as e:
    FAISS_AVAILABLE = False
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"import-check","hypothesisId":"H1,H2,H3","location":"rag_handler.py:20","message":"faiss导入失败","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000)}) + "\n")
    except: pass
    # #endregion
    import warnings
    warnings.warn(
        f"faiss模块未安装。RAG功能将使用替代方案（基于numpy的相似度计算）。"
        f"要启用完整RAG功能，请运行: pip install faiss-cpu"
    )

class RAGHandler:
    """RAG因子启发处理器"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.8):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.index = None
        self.factor_data = []
        self.factor_embeddings = []
        
    def build_index(self, factors: List['Factor']):
        """构建因子索引"""
        self.factor_data = factors
        
        # 生成嵌入向量
        texts = [
            f"{factor.expr} | {factor.explanation} | IC: {factor.ic_mean:.4f}"
            for factor in factors
        ]
        
        self.factor_embeddings = self.embedding_model.encode(texts)
        
        # 构建FAISS索引（如果可用）或使用numpy替代方案
        if FAISS_AVAILABLE:
            dimension = self.factor_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
            self.index.add(self.factor_embeddings.astype(np.float32))
        else:
            # 使用numpy作为替代方案
            self.index = None  # 标记为使用numpy方案
    
    def search(self, query: str, top_k: int = 5) -> List['Factor']:
        """搜索相似因子"""
        if len(self.factor_data) == 0:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        if FAISS_AVAILABLE and self.index is not None:
            # 使用FAISS搜索
            distances, indices = self.index.search(
                query_embedding.astype(np.float32).reshape(1, -1), 
                min(top_k, len(self.factor_data))
            )
            
            # 过滤并返回结果
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if dist >= self.similarity_threshold and idx < len(self.factor_data):
                    results.append(self.factor_data[idx])
            
            return results
        else:
            # 使用numpy计算相似度（替代方案）
            if len(self.factor_embeddings) == 0:
                return []
            
            # 计算余弦相似度
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return []
            
            # 归一化查询向量
            query_normalized = query_embedding / query_norm
            
            # 归一化所有因子嵌入
            factor_norms = np.linalg.norm(self.factor_embeddings, axis=1, keepdims=True)
            factor_norms[factor_norms == 0] = 1  # 避免除零
            factor_normalized = self.factor_embeddings / factor_norms
            
            # 计算内积（余弦相似度）
            similarities = np.dot(factor_normalized, query_normalized)
            
            # 获取top_k
            top_k_actual = min(top_k, len(self.factor_data))
            top_indices = np.argsort(similarities)[::-1][:top_k_actual]
            
            # 过滤并返回结果
            results = []
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    results.append(self.factor_data[idx])
            
            return results
    
    def add_factors(self, factors: List['Factor']):
        """添加新因子到索引"""
        if not factors:
            return
        
        # 添加数据
        self.factor_data.extend(factors)
        
        # 生成新嵌入
        new_texts = [
            f"{factor.expr} | {factor.explanation} | IC: {factor.ic_mean:.4f}"
            for factor in factors
        ]
        new_embeddings = self.embedding_model.encode(new_texts)
        
        # 添加到索引
        if FAISS_AVAILABLE:
            if self.index is None:
                dimension = new_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                # 如果有现有嵌入，先添加它们
                if len(self.factor_embeddings) > 0:
                    self.index.add(self.factor_embeddings.astype(np.float32))
            
            self.factor_embeddings = np.vstack([self.factor_embeddings, new_embeddings]) if len(self.factor_embeddings) > 0 else new_embeddings
            self.index.add(new_embeddings.astype(np.float32))
        else:
            # numpy方案：直接添加到嵌入列表
            if len(self.factor_embeddings) > 0:
                self.factor_embeddings = np.vstack([self.factor_embeddings, new_embeddings])
            else:
                self.factor_embeddings = new_embeddings
    
    def save(self, path: str):
        """保存RAG索引"""
        save_data = {
            'factor_data': self.factor_data,
            'embeddings': self.factor_embeddings,
            'index': None
        }
        
        if FAISS_AVAILABLE and self.index is not None:
            save_data['index'] = faiss.serialize_index(self.index)
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str):
        """加载RAG索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.factor_data = data['factor_data']
        self.factor_embeddings = data['embeddings']
        
        if FAISS_AVAILABLE and data.get('index'):
            self.index = faiss.deserialize_index(data['index'])
        else:
            self.index = None