# src/__init__.py

__version__ = "1.0.0"

from src.modules.adapters.chinese import ChineseAdapter
from src.modules.adapters.english import EnglishAdapter

from src.modules.subnets.lexical import LexicalSubnet
from src.modules.subnets.syntax import SyntaxSubnet
from src.modules.subnets.context import ContextSubnet
from src.modules.subnets.domain import DomainSubnet

from src.modules.coordinators.attention_coordinator import AttentionCoordinator