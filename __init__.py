"""
虚假视频检测工作流包
不使用crewai，完全基于自定义代理实现
"""

from .workflows.detector import FakeVideoDetectorWorkflow, kickoff, batch_kickoff
from .agents import (
    BaseAgent,
    ConsistencyAnalyzer,
    AIDetector,
    OffensiveLanguageDetector,
    FactChecker,
    Retriever,
    Locator,
    Integrator,
)

__version__ = "1.0.0"
__all__ = [
    'FakeVideoDetectorWorkflow',
    'kickoff',
    "batch_kickoff",
    'BaseAgent',
    'ConsistencyAnalyzer',
    'AIDetector',
    'OffensiveLanguageDetector',
    'FactChecker',
    'Retriever',
    'Locator',
    'Integrator',
]
