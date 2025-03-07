"""
This file exports policy models to use during inference / self-play.
"""

import os
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from models.ac_models_hetero import Esc1, Esc2, Fight1, Fight2

# 实验配置参数设置
# LEVEL: 当前关卡等级，用于生成实验目录名
# MODE: 实验模式（战斗/逃跑），决定使用的模型类型
#define experiment folder name
LEVEL = 3
MODE = 'fight'
# 生成实验目录名称，格式：LevelX_mode
EXP_DIR = f'Level{LEVEL}_{MODE}'

#define policy folder name
# 策略模型最终保存目录
POL_DIR = 'policies'

# 根据模式注册对应的自定义模型
if MODE == "fight":
    # 注册战斗模式下的双智能体模型
    ModelCatalog.register_custom_model(f"ac1_model",Fight1)
    ModelCatalog.register_custom_model(f"ac2_model",Fight2)
else:
    # 注册逃跑模式下的双智能体模型（带_esc后缀）
    ModelCatalog.register_custom_model(f"ac1_model_esc",Esc1)
    ModelCatalog.register_custom_model(f"ac2_model_esc",Esc2)

# 策略模型导出流程（处理两个智能体）
for i in range(1,3):
    # 构建检查点路径：从实验结果目录加载训练好的策略
    check = os.path.join(os.path.dirname(__file__), 'results', EXP_DIR, 'checkpoint', 'policies', f'ac{i}_policy')
    # 从检查点加载策略并导出为可部署格式
    pol = Policy.from_checkpoint(check)
    save_dir = os.path.join(os.path.dirname(__file__), POL_DIR)
    # 导出模型到指定目录
    pol.export_model(save_dir)
    
    # 重命名模型文件（包含关卡等级、智能体编号和模式信息）
    policy_name = f'L{LEVEL}_AC{i}_{MODE}'
    os.rename(f'{POL_DIR}/model.pt', f'{POL_DIR}/{policy_name}.pt')

# 输出最终结果提示
print(f"{MODE} policies exported to folder {POL_DIR}")