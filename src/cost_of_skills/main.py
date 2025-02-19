import torch
import numpy as np
from typing import List, Dict, Tuple, Union
import pandas as pd
import json

class IntervalSkillsModel:
    """
    Модель для оценки стоимости навыков на основе интервальных зарплатных данных.
    """
    
    def __init__(
        self,
        lr: float = 5.0,
        num_iterations: int = 10000,
        initial_value: float = 10000.0,
        device: Union[str, torch.device] = None
    ):
        # ... существующий код инициализации ...
        
    # ... существующие методы ...
    
    def save_to_json(self, filepath: str) -> None:
        """
        Сохраняет модель в JSON файл.
        
        Args:
            filepath: путь для сохранения JSON файла
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        # Создаем словарь с параметрами модели
        model_data = {
            "model_params": {
                "lr": self.lr,
                "num_iterations": self.num_iterations,
                "initial_value": self.initial_value
            },
            "skill_costs": {},
            "skill_id_dict": self.skill_id_dict
        }
        
        # Добавляем стоимости навыков
        with torch.no_grad():
            for skill, idx in self.skill_id_dict.items():
                model_data["skill_costs"][skill] = float(self.x[idx].cpu())
        
        # Сохраняем в JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load_from_json(cls, filepath: str) -> 'IntervalSkillsModel':
        """
        Загружает модель из JSON файла.
        
        Args:
            filepath: путь к JSON файлу
            
        Returns:
            IntervalSkillsModel: загруженная модель
        """
        # Загружаем данные из JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        # Создаем новый экземпляр модели
        model = cls(
            lr=model_data["model_params"]["lr"],
            num_iterations=model_data["model_params"]["num_iterations"],
            initial_value=model_data["model_params"]["initial_value"]
        )
        
        # Восстанавливаем словарь навыков
        model.skill_id_dict = model_data["skill_id_dict"]
        
        # Восстанавливаем веса модели
        weights = torch.zeros(len(model.skill_id_dict), device=model.device)
        for skill, cost in model_data["skill_costs"].items():
            idx = model.skill_id_dict[skill]
            weights[idx] = cost
            
        model.x = torch.nn.Parameter(weights)
        model.is_fitted = True
        
        return model
    
    def save_skill_costs_json(self, filepath: str) -> None:
        """
        Сохраняет только стоимости навыков в JSON файл (для Gradio).
        
        Args:
            filepath: путь для сохранения JSON файла
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving skill costs")
            
        skill_costs = self.get_skill_costs()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(skill_costs, f, ensure_ascii=False, indent=2)
