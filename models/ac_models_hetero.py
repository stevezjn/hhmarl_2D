from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import torch.nn.functional as F
from ray.rllib.policy.rnn_sequencing import add_time_dimension
torch, nn = try_import_torch()

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3

OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

SS_AGENT_AC1 = 12
SS_AGENT_AC2 = 10

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

class Esc1(TorchModelV2, nn.Module):
    """强化学习环境中的策略网络模型，继承自TorchModelV2和nn.Module
    
    Args:
        observation_space (gym.Space): 环境观测空间
        action_space (gym.Space): 动作输出空间
        num_outputs (int): 策略网络输出维度
        model_config (dict): 模型配置字典
        name (str): 模型名称
    """
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        # 双继承初始化（RLlib框架要求）
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # 网络结构定义
        # 共享网络层（需外部定义）
        self.shared_layer = SHARED_LAYER

        # 初始化输入缓存
        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        # 价值函数输入缓存
        self._v1 = None

        # 观测特征编码分支
        # 处理前7维观测特征
        self.inp1 = SlimFC(
            7,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        # 处理中间18维观测特征
        self.inp2 = SlimFC(
            18,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        # 处理后5维观测特征
        self.inp3 = SlimFC(
            5,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        # 动作输出分支
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

        # 价值函数计算分支
        # 拼接观测和动作的编码层
        self.inp1_val = SlimFC(
            OBS_ESC_AC1+ACTION_DIM_AC1+OBS_ESC_AC2+ACTION_DIM_AC2,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        # 价值函数输出层
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """前向传播过程
        
        Args:
            input_dict (dict): 包含观测数据的字典，结构为：
                obs_1_own: 智能体自身观测特征（形状为[:,30]）
                obs_2: 其他智能体观测数据
                act_1_own: 自身历史动作
                act_2: 其他智能体动作
            state: RNN隐藏状态（本实现未使用）
            seq_lens: 序列长度（本实现未使用）

        Returns:
            tuple: (动作输出张量, 空状态列表)
        """
        # 特征拆分与编码
        # 提取前7维特征
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:7]
        # 提取中间18维特征
        self._inp2 = input_dict["obs"]["obs_1_own"][:,7:25]
        # 提取最后5维特征
        self._inp3 = input_dict["obs"]["obs_1_own"][:,25:]
        # 价值函数输入拼接（观测+动作）
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)

        # 多分支特征融合
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        # 通过共享层
        x = self.shared_layer(x)
        x = self.act_out(x)
        # 返回动作输出和空状态
        return x, []

    @override(ModelV2)
    def value_function(self):
        """计算状态价值函数
        
        Returns:
            Tensor: 形状为[batch_size, 1]的价值估计
            
        Raises:
            AssertionError: 前向传播未执行时调用会触发异常
        """
        assert self._v1 is not None, "must call forward first!"
        # 编码拼接特征
        x = self.inp1_val(self._v1)
        # 共享层处理
        x = self.shared_layer(x)
        x = self.val_out(x)
        # 输出价值估计
        return torch.reshape(x, [-1])
    
class Esc2(TorchModelV2, nn.Module):
    """
    逃脱策略专用神经网络模型，实现多智能体协同决策的actor-critic架构

    继承结构:
        TorchModelV2 - Ray RLlib框架的PyTorch模型基类
        nn.Module   - PyTorch神经网络模块基类

    网络架构特性:
        - 三路特征分流处理：将观测空间分解为3个语义子空间独立编码
        - 参数共享机制：策略网络与值函数网络共享中间层（shared_layer）
        - 正交初始化：所有全连接层使用正交初始化保证训练稳定性

    输入特征结构（obs_1_own）:
        [0:6]    : 自身基础状态（坐标、航向、速度等）
        [6:24]   : 环境交互特征（威胁方位、友军状态等）
        [24:]    : 装备状态（燃油、武器剩余等）
    """
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        """TorchModelV2和nn.Module的双继承初始化构造函数
        
        参数说明：
            observation_space (gym.Space): 环境观测空间定义
            action_space (gym.Space): 动作空间定义
            num_outputs (int): 模型最终输出的维度
            model_config (dict): 模型配置参数字典
            name (str): 模型实例名称标识

        结构说明：
            1. 定义三个输入分支(inp1/inp2/inp3)的全连接层
            2. 动作输出层(act_out)和价值输出层(val_out)
            3. 中间状态占位符(_inp1/_inp2/_inp3/_v1)
        """
        # 双继承初始化（必须显式调用两个父类构造器）
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # 共享层引用（具体实现应在其他地方定义）
        self.shared_layer = SHARED_LAYER

        # 中间状态初始化占位符
        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None

        # 定义三个输入分支的全连接层
        # inp1: 6维输入 -> 150维隐藏层(tanh激活)
        # inp2: 18维输入 -> 250维隐藏层(tanh激活) 
        # inp3: 5维输入 -> 100维隐藏层(tanh激活)
        self.inp1 = SlimFC(
            6,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.inp2 = SlimFC(
            18,
            250,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.inp3 = SlimFC(
            5,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        # 动作输出层（500维输入->num_outputs维输出，无激活）
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

        # 价值函数计算分支
        # 输入维度由多个常量组合而成(OBS_ESC_AC1+ACTION_DIM_AC1+...)
        self.inp1_val = SlimFC(
            OBS_ESC_AC1+ACTION_DIM_AC1+OBS_ESC_AC2+ACTION_DIM_AC2,
            500,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        # 价值输出层（500维输入->标量输出，无激活）
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        重写ModelV2的前向传播方法
        
        参数：
            input_dict (dict): 输入数据字典，包含观测和动作数据。结构：
                - obs: 
                    - obs_1_own: 主体1的观测数据，前6维为基本状态，中间18维为传感器数据，后续为其他特征
                    - act_1_own: 主体1的当前动作
                    - obs_2: 主体2的观测数据
                    - act_2: 主体2的动作
            state (list): RNN的隐藏状态（本实现未使用RNN）
            seq_lens (Tensor): 序列长度数据（本实现未使用序列数据）
            
        返回：
            tuple: 包含：
                - x (Tensor): 经过神经网络处理后的输出张量
                - []: 空状态列表（保持接口兼容性）
        """
        
        # 特征分解：将主体1的观测拆分为三个子特征组
        # 基本状态特征
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:6]
        # 传感器特征
        self._inp2 = input_dict["obs"]["obs_1_own"][:,6:24]
        # 扩展特征
        self._inp3 = input_dict["obs"]["obs_1_own"][:,24:]
        # 联合特征拼接（当前未使用但保留的代码结构）
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)

        # 多分支特征处理与融合
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        # 共享层处理与最终输出
        # 通过共享的全连接层
        x = self.shared_layer(x)
        # 激活函数输出层
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        """计算状态值的神经网络前向传播过程

        通过三层网络结构处理输入特征，最终输出状态值的标量估计。
        要求调用前必须执行过forward()方法完成特征计算。

        Returns:
            torch.Tensor: 形状为[batch_size]的一维张量，包含每个样本的状态值估计
            
        Raises:
            AssertionError: 若未预先调用forward方法初始化_v1特征时抛出
        """
        # 前置条件校验：确保已完成特征提取
        assert self._v1 is not None, "must call forward first!"
        # 特征处理流程：
        # 1. 通过输入值转换层 2.共享层处理 3.输出层计算
        x = self.inp1_val(self._v1)
        x = self.shared_layer(x)
        x = self.val_out(x)
        # 将输出转换为(batch_size,)形状的一维张量
        return torch.reshape(x, [-1])

class Fight1(RecurrentNetwork, nn.Module):
    """
    本模型专为战场环境下的多智能体协同决策设计，主要功能包括：
    - 策略网络（Actor）：生成主智能体的动作概率分布
    - 价值网络（Critic）：评估战场全局状态价值
    采用双流注意力机制处理时空特征，适用于动态战场环境下的决策问题

    Args:
        observation_space (gym.Space): 环境观测空间定义
        action_space (gym.Space): 动作空间定义
        num_outputs (int): 策略网络输出维度（对应动作空间维度）
        model_config (dict): 模型配置字典
        name (str): 模型实例名称

    Attributes:
        shared_layer (nn.Module): 策略/价值共享的特征提取层
        att_act (nn.MultiheadAttention): 策略分支的时空注意力层
        att_val (nn.MultiheadAttention): 价值分支的协同注意力层
        _val (Tensor): 当前批次的状态价值估计缓存
        batch_len (int): 当前处理的批次大小记录

    Methods:
        forward: 执行前向传播，返回动作logits和RNN状态
        value_function: 获取最终的状态价值估计
        get_initial_state: 生成RNN初始状态

    模型特性：
    - 双模态处理：分离处理个体状态特征（坐标、速度）和环境交互特征（威胁感知）
    - 协同注意力：显式建模智能体间的动作影响关系
    - 时序残差连接：通过时间维度注意力保留关键战场事件记忆
    - 正交初始化：所有全连接层使用正交权重初始化策略

    输入数据结构要求：
    输入字典应包含以下嵌套结构：
    {
        "obs": {
            "obs_1_own": Tensor  # 主智能体观测 [batch_size, OBS_AC1]
                [:SS_AGENT_AC1]   个体状态特征（坐标、速度、姿态等）
                [SS_AGENT_AC1:]   战场环境特征（威胁方位、友军状态等）
            "obs_2": Tensor      # 协作智能体观测 [batch_size, OBS_AC2]
            "act_1_own": Tensor  # 主智能体历史动作 [batch_size, ACTION_DIM_AC1]
            "act_2": Tensor       # 协作智能体动作 [batch_size, ACTION_DIM_AC2]
        }
    }
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        """神经网络模型的初始化构造函数
        
        用于构建包含注意力机制和全连接层的深度强化学习模型，支持多智能体协同场景

        Parameters:
            observation_space (gym.Space): 环境的观察空间定义
            action_space (gym.Space): 代理可执行的动作空间
            num_outputs (int): 策略网络输出层的维度大小
            model_config (dict): 模型结构配置参数字典
            name (str): 模型实例的标识名称
        """
        # 初始化父类神经网络模块
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        # 定义中间变量存储空间
        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        # 价值网络中间状态容器
        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._val = None

        # 批处理长度记录器
        self.batch_len = 0

        self.shared_layer = SHARED_LAYER
        """注意力机制层定义"""
        # 动作特征多头注意力层（100维特征，2个头）
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        # 价值特征多头注意力层（150维特征，2个头）
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)

        """策略网络全连接层定义"""
        # 输入处理分支1：状态特征映射到200维
        self.inp1 = SlimFC(
            SS_AGENT_AC1,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 输入处理分支2：剩余观测特征映射到200维
        self.inp2 = SlimFC(
            OBS_AC1-SS_AGENT_AC1,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 综合输入处理层：合并特征到100维
        self.inp3 = SlimFC(
            OBS_AC1,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 策略输出层：映射到动作空间维度
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )
        """价值网络全连接层定义"""
        # 价值分支1：状态+动作联合特征处理
        self.v1 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 价值分支2：扩展观测特征处理
        self.v2 = SlimFC(
            OBS_AC2+ACTION_DIM_AC2,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 价值分支3：多源特征融合层
        self.v3 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1+OBS_AC2+ACTION_DIM_AC2,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 价值输出层：最终价值评估
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        """        
        获取循环神经网络的初始状态

        本方法重写父类RecurrentNetwork的get_initial_state方法，用于生成并返回网络
        的初始状态。初始状态通常用于循环神经网络(RNN/LSTM/GRU)的时间步计算初始化

        Args:
            self: 当前类实例的引用，自动传入无需显式传递

        Returns:
            list[torch.Tensor]: 包含单个零张量的列表，该张量形状为(1,)，
                表示网络的初始状态值。列表结构为后续可能的多层状态预留扩展空间
        """
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """执行前向传播计算，处理多模态输入并融合注意力特征
        
        Args:
            input_dict: 包含观测和动作输入的字典。必须包含键：
                "obs" -> 包含各智能体观测数据的字典，结构为：
                    "obs_1_own": 主智能体自身观测数据
                    "act_1_own": 主智能体自身动作数据
                    "obs_2": 其他智能体观测数据
                    "act_2": 其他智能体动作数据
            state: RNN隐藏状态（当前实现未使用）
            seq_lens: 序列长度张量，用于时间维度处理
            
        Returns:
            tuple: 包含处理后的特征张量和空列表（保持接口兼容）
        """
        
        # 输入数据预处理
        # 分解主智能体观测数据的前SS_AGENT_AC1维和后部分
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT_AC1]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT_AC1:]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]

        # 注意力机制输入构建
        # 拼接主智能体观测+动作和其他智能体观测+动作
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((self._v1, self._v2), dim=1)

        # 主特征处理流程
        # 融合分解后的输入特征并通过注意力机制
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        # 特征融合与输出
        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        # 价值网络处理流程
        # 处理拼接后的多智能体信息并计算价值估计
        y = torch.cat((self.v1(self._v1),self.v2(self._v2)),dim=1)
        y_full = self.v3(self._v3)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = nn.functional.normalize(y_full + y_att.reshape((self.batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        """
        获取当前模型的状态值函数估计

        返回:
            torch.Tensor: 经过重塑后的值函数输出张量，形状为[-1]的一维张量。
                        表示环境状态的价值评估

        说明:
            - 需要先调用forward()方法计算结果
            - 会对self._val进行维度重塑操作
        """
        # 验证前向计算结果是否存在
        assert self._val is not None, "must call forward first!"
        # 将值函数输出重塑为1D张量
        return torch.reshape(self._val, [-1])

class Fight2(RecurrentNetwork, nn.Module):
    """
    本模型结合循环网络和全连接网络，实现：
    - 策略网络（Actor）：生成动作概率分布
    - 价值网络（Critic）：评估状态价值

    Args:
        observation_space (gym.Space): 环境观测空间
        action_space (gym.Space): 动作空间
        num_outputs (int): 策略网络输出维度（对应动作空间维度）
        model_config (dict): 模型配置字典
        name (str): 模型名称

    Attributes:
        shared_layer (nn.Module): 策略/价值网络共享的全连接层
        att_act (nn.MultiheadAttention): 策略分支的时序注意力机制
        att_val (nn.MultiheadAttention): 价值分支的时序注意力机制
        _val (Tensor): 当前批次的状态价值估计缓存
        batch_len (int): 当前处理的批次大小

    Methods:
        forward: 前向传播，返回动作logits和RNN状态
        value_function: 获取最后计算的状态价值估计
        get_initial_state: 获取RNN初始状态

    主要特点：
    - 双分支结构：分别处理策略生成和价值估计
    - 注意力增强：使用MultiheadAttention捕捉时序依赖
    - 残差连接：保持梯度流动的同时增强特征表达
    - 参数共享：策略/价值网络共享中间层参数
    - 多源输入：处理自身观测、其他智能体观测及历史动作信息

    输入数据约定：
    输入字典应包含以下结构：
    {
        "obs": {
            "obs_1_own": Tensor,  # 自身观测（分为两部分处理）
            "obs_2": Tensor,      # 其他智能体观测
            "act_1_own": Tensor,  # 自身历史动作
            "act_2": Tensor       # 其他智能体历史动作
        }
    }
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        """Initialize the neural network model for reinforcement learning agent.

        Args:
            observation_space (gym.Space): Environment's observation space
            action_space (gym.Space): Environment's action space
            num_outputs (int): Dimension of the policy output layer
            model_config (dict): Configuration dictionary for model architecture
            name (str): Name identifier for the model

        Returns:
            None
        """
        # 基础模块初始化
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        # 初始化中间层输入/值函数占位符
        self._inp1 = None
        self._inp2 = None
        self._inp3 = None

        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._val = None

        # 批处理长度记录
        self.batch_len = 0

        """注意力机制模块"""
        # 多头注意力层：动作分支(100维特征/2头) 值分支(150维特征/2头)
        self.shared_layer = SHARED_LAYER
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)

        """动作分支网络结构"""
        # 输入处理模块（正交初始化 + Tanh激活）
        self.inp1 = SlimFC(
            SS_AGENT_AC2,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            OBS_AC2-SS_AGENT_AC2,
            200,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            OBS_AC2,
            100,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 动作输出层（无激活函数）
        self.act_out = SlimFC(
            500,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )
        """价值分支网络结构"""
        # 价值特征融合模块
        self.v1 = SlimFC(
            OBS_AC2+ACTION_DIM_AC2,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v2 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1,
            175,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        self.v3 = SlimFC(
            OBS_AC1+ACTION_DIM_AC1+OBS_AC2+ACTION_DIM_AC2,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_,
        )
        # 价值输出层（标量输出）
        self.val_out = SlimFC(
            500,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        """获取循环神经网络的初始隐藏状态
        
        覆盖父类方法，返回由零张量构成的初始状态列表。
        该状态用于初始化循环神经网络的时间步记忆
        
        Returns:
            list[torch.Tensor]: 包含单个零张量的列表，张量形状为(1,)，
                表示批次大小为1的初始隐藏状态
        """
        return [torch.zeros(1)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """执行前向传播计算，处理观测数据并生成网络输出

        Args:
            input_dict (dict): 输入观测字典，应包含：
                - obs_1_own: 自身智能体观察数据（包含需要切分的特征）
                - act_1_own: 自身智能体动作数据
                - obs_2/act_2: 其他智能体观察和动作数据
            state (Tensor): RNN网络状态（未使用但保留接口）
            seq_lens (Tensor): 序列长度数据，用于时间维度处理

        Returns:
            Tuple[Tensor, List]: 包含策略输出和空状态列表的元组
        """

        # ==================== 输入分解处理 ====================
        # 切分主智能体的观测特征到不同子网络
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:SS_AGENT_AC2]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,SS_AGENT_AC2:]
        self._inp3 = input_dict["obs"]["obs_1_own"]
        self.batch_len = self._inp1.shape[0]

        # 构建价值网络的多源输入组合
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((self._v1, self._v2), dim=1)

        # ==================== 主特征处理分支 ====================
        # 处理切分后的主特征并融合
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2)),dim=1) 
        x_full = self.inp3(self._inp3)
        # 时序注意力增强处理
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = nn.functional.normalize(x_full + x_att.reshape((self.batch_len, -1)))

        # 特征最终融合与输出
        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        # ==================== 价值估计分支 ====================
        # 处理多源输入的价值特征

        y = torch.cat((self.v1(self._v1),self.v2(self._v2)),dim=1)
        y_full = self.v3(self._v3)
        # 时序注意力增强处理（价值分支）
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens, framework="torch", time_major=False)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = nn.functional.normalize(y_full + y_att.reshape((self.batch_len, -1)))

        # 价值估计最终计算
        y = torch.cat((y, y_full), dim=1)
        y = self.shared_layer(y)
        self._val = self.val_out(y)

        return x, []
    
    @override(ModelV2)
    def value_function(self):
        """获取当前模型的状态价值估计值。

        继承并覆盖ModelV2的value_function方法，用于返回经过reshape处理的价值函数输出。
        该方法需要先执行forward前向计算以获取_val值。

        Returns:
            torch.Tensor: 重塑为二维形状[-1,1]的一维价值张量，
                表示每个样本对应的状态价值估计

        Raises:
            AssertionError: 若未先执行forward方法初始化_val值则抛出异常
        """
        # 前置校验确保已执行forward计算
        assert self._val is not None, "must call forward first!"
        
        # 将价值张量重塑为二维格式（保持第0维自动计算）
        return torch.reshape(self._val, [-1])
