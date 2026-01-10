# ==================== factor/synthesizer.py ====================
"""
因子合成器
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .factor import Factor

class FactorSynthesizer:
    """因子合成器"""
    
    @staticmethod
    def synthesize_price_factors(factors: List[Factor], 
                               method: str = 'ic_weighted') -> Dict[str, Any]:
        """合成量价因子"""
        if not factors:
            return {}
        
        # 对齐所有因子的时间
        aligned_factors = FactorSynthesizer._align_factors(factors)
        
        # 选择合成方法
        if method == 'ic_weighted':
            return FactorSynthesizer._ic_weighted_synthesis(factors, aligned_factors)
        elif method == 'equal_weighted':
            return FactorSynthesizer._equal_weighted_synthesis(aligned_factors)
        elif method == 'pca':
            return FactorSynthesizer._pca_synthesis(aligned_factors)
        else:
            return FactorSynthesizer._ic_weighted_synthesis(factors, aligned_factors)
    
    @staticmethod
    def synthesize_fundamental_factors(factors: List[Factor]) -> Dict[str, Any]:
        """合成基本面因子"""
        return FactorSynthesizer.synthesize_price_factors(factors, 'equal_weighted')
    
    @staticmethod
    def _align_factors(factors: List[Factor]) -> List[pd.DataFrame]:
        """对齐因子时间序列"""
        # 获取共同的时间索引和股票代码
        common_index = factors[0].values.index
        common_columns = factors[0].values.columns
        
        for factor in factors[1:]:
            common_index = common_index.intersection(factor.values.index)
            common_columns = common_columns.intersection(factor.values.columns)
        
        # 对齐所有因子
        aligned = []
        for factor in factors:
            aligned_values = factor.values.loc[common_index, common_columns]
            # 截面标准化
            aligned_values = aligned_values.apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1
            )
            aligned.append(aligned_values)
        
        return aligned
    
    @staticmethod
    def _ic_weighted_synthesis(factors: List[Factor], aligned_factors: List[pd.DataFrame]) -> Dict[str, Any]:
        """IC加权合成"""
        # 获取每个因子的IC均值作为权重
        ic_weights = [factor.ic_mean for factor in factors]
        ic_weights = np.abs(ic_weights) / np.sum(np.abs(ic_weights))
        
        # 加权合成
        synthesized = pd.DataFrame(0, 
                                 index=aligned_factors[0].index,
                                 columns=aligned_factors[0].columns)
        
        for i, (weight, factor_values) in enumerate(zip(ic_weights, aligned_factors)):
            synthesized += weight * factor_values
        
        return {
            'synthesized_values': synthesized,
            'weights': ic_weights,
            'method': 'ic_weighted'
        }
    
    @staticmethod
    def _equal_weighted_synthesis(aligned_factors: List[pd.DataFrame]) -> Dict[str, Any]:
        """等权合成"""
        synthesized = pd.DataFrame(0,
                                 index=aligned_factors[0].index,
                                 columns=aligned_factors[0].columns)
        
        for factor_values in aligned_factors:
            synthesized += factor_values
        
        synthesized /= len(aligned_factors)
        
        return {
            'synthesized_values': synthesized,
            'weights': [1/len(aligned_factors)] * len(aligned_factors),
            'method': 'equal_weighted'
        }
    
    @staticmethod
    def _pca_synthesis(aligned_factors: List[pd.DataFrame]) -> Dict[str, Any]:
        """PCA合成"""
        # 将因子值展平为特征矩阵
        n_factors = len(aligned_factors)
        n_times = len(aligned_factors[0].index)
        n_stocks = len(aligned_factors[0].columns)
        
        # 创建特征矩阵
        features = np.zeros((n_times * n_stocks, n_factors))
        
        for i in range(n_factors):
            features[:, i] = aligned_factors[i].values.flatten()
        
        # PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        synthesized_flat = pca.fit_transform(features)[:, 0]
        
        # 重塑为DataFrame
        synthesized = pd.DataFrame(
            synthesized_flat.reshape(n_times, n_stocks),
            index=aligned_factors[0].index,
            columns=aligned_factors[0].columns
        )
        
        return {
            'synthesized_values': synthesized,
            'pca_components': pca.components_[0],
            'explained_variance': pca.explained_variance_ratio_[0],
            'method': 'pca'
        }