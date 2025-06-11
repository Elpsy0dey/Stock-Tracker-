"""
Strategy Manager for handling trading strategy documents
"""
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class StrategyManager:
    def __init__(self, strategy_dir: str = "strategies"):
        """
        Initialize the Strategy Manager
        
        Args:
            strategy_dir: Directory containing strategy documents
        """
        self.strategy_dir = Path(strategy_dir)
        self.strategy_dir.mkdir(exist_ok=True)
        self._current_strategy = None
        self._last_modified = None
        
    def save_strategy(self, content: str, name: str = "current_strategy.txt") -> None:
        """
        Save a strategy document
        
        Args:
            content: The strategy document content
            name: Name of the strategy file
        """
        file_path = self.strategy_dir / name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        self._current_strategy = None  # Reset cache
        self._last_modified = None
        
    def get_strategy(self, name: str = "current_strategy.txt") -> str:
        """
        Get the current strategy document
        
        Args:
            name: Name of the strategy file to read
            
        Returns:
            str: The strategy document content
        """
        file_path = self.strategy_dir / name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy document {name} not found")
            
        # Check if we need to reload the strategy
        current_mtime = os.path.getmtime(file_path)
        if (self._current_strategy is None or 
            self._last_modified is None or 
            current_mtime > self._last_modified):
            
            with open(file_path, 'r', encoding='utf-8') as f:
                self._current_strategy = f.read()
            self._last_modified = current_mtime
            
        return self._current_strategy
        
    def list_strategies(self) -> List[str]:
        """
        List all available strategy documents
        
        Returns:
            List[str]: List of strategy document names
        """
        return [f.name for f in self.strategy_dir.glob("*.txt")]
        
    def get_strategy_metadata(self, name: str = "current_strategy.txt") -> Dict:
        """
        Get metadata about a strategy document
        
        Args:
            name: Name of the strategy file
            
        Returns:
            Dict: Metadata including last modified time and size
        """
        file_path = self.strategy_dir / name
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy document {name} not found")
            
        stats = file_path.stat()
        return {
            "name": name,
            "last_modified": datetime.fromtimestamp(stats.st_mtime),
            "size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime)
        } 