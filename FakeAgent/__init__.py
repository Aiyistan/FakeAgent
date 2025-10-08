"""
虚假视频检测工作流
"""

from .workflow import FakeVideoDetectorWorkflow, kickoff
from .agents import (
    BaseAgent,
    ConsistencyAnalyzer,
    AIDetector,
    OffensiveLanguageDetector,
    FactChecker,
    Retriever,
    Locator,
    Integrator,
    # Answer
)

__version__ = "1.0.0"
__all__ = [
    'FakeVideoDetectorWorkflow',
    'kickoff',
    'BaseAgent',
    'ConsistencyAnalyzer',
    'AIDetector',
    'OffensiveLanguageDetector',
    'FactChecker',
    'Retriever',
    'Locator',
    'Integrator',
    # 'Answer'
]
