import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.rnn_sequencing import add_time_dimension
torch, nn = try_import_torch()

N_OPP_HL = 2 #change for sensing
OBS_OPP = 10
OBS_DIM = 14+OBS_OPP*N_OPP_HL

SHARED_LAYER = SlimFC(
    500,
    500,
    activation_fn= nn.Tanh,
    initializer=torch.nn.init.orthogonal_
)

class CommanderGru(RecurrentNetwork, nn.Module):
    """基于GRU的强化学习策略网络，用于处理序列决策任务

    继承自RecurrentNetwork和nn.Module，包含动作生成和价值评估双路径结构

    Args:
        observation_space (gym.Space): 环境观察空间定义
        action_space (gym.Space): 环境动作空间定义
        num_outputs (int): 输出动作的维度
        model_config (dict): 模型配置参数字典
        name (str): 模型名称标识
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        """神经网络的初始化构造函数
        
        参数说明:
            observation_space (gym.Space): 环境观测空间定义
            action_space (gym.Space): 环境动作空间定义
            num_outputs (int): 策略网络输出的动作维度
            model_config (dict): 模型配置参数字典
            name (str): 模型名称标识
            
        功能:
            初始化包含GRU和全连接层的双分支网络结构，包含：
            - 动作分支网络（rnn_act + act_out）
            - 价值评估分支网络（rnn_val + val_out）
        """
        # 基础模块初始化
        nn.Module.__init__(self)
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        # 初始化中间状态缓存变量（用于序列数据处理）
        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._val = None

        # 共享层定义（具体实现依赖外部SHARED_LAYER定义）
        self.shared_layer = SHARED_LAYER
        # 动作分支的GRU层（处理200维特征的时间序列）
        self.rnn_act = nn.GRU(200, 200, batch_first=True)
        # 价值分支的GRU层（处理200维特征的时间序列） 
        self.rnn_val = nn.GRU(200, 200, batch_first=True)

        # 动作网络输入处理分支（4个并行输入处理通道）
        # 输入维度分别为4/N_OPP_HL*OBS_OPP/10/OBS_DIM，输出维度50/200/50/200
        self.inp1 = SlimFC(
            4,50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp2 = SlimFC(
            N_OPP_HL*OBS_OPP,200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp3 = SlimFC(
            10,50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.inp4 = SlimFC(
            OBS_DIM,200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        # 动作输出层（500维特征到动作空间）
        self.act_out = SlimFC(
            500,num_outputs, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )
        

        # 价值网络输入处理分支（3个并行输入通道+融合层）
        # 每个通道处理OBS_DIM+1维输入，输出100维特征
        self.v1 = SlimFC(
            OBS_DIM+1,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v2 = SlimFC(
            OBS_DIM+1,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        self.v3 = SlimFC(
            OBS_DIM+1,100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        # 特征融合层（将3个100维特征合并为200维）
        self.v4 = SlimFC(
            3*(OBS_DIM+1),200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_,
        )
        # 价值输出层（500维特征到标量价值）
        self.val_out = SlimFC(
            500,1, activation_fn=None, initializer=torch.nn.init.orthogonal_,
        )

    @override(RecurrentNetwork)
    def get_initial_state(self):
        """
        获取模型的初始隐藏状态
        
        覆盖父类方法，返回符合模型结构要求的初始状态张量。
        适用于具有双层隐藏结构的循环神经网络（如LSTM/GRU）
        
        Returns:
            list[torch.Tensor, torch.Tensor]: 
                包含两个零值张量的列表，每个张量形状为(200,)
                第一个张量对应第一个隐藏层的初始状态
                第二个张量对应第二个隐藏层的初始状态
        """
        return [torch.zeros(200), torch.zeros(200)]
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """执行神经网络前向传播
        
        Args:
            input_dict (dict): 输入数据字典，包含观测和动作数据。结构：
                - obs: 
                    - obs_1_own: 本体观测张量 [batch, obs_dim]
                    - act_1_own: 本体动作张量
                    - obs_2/3: 其他智能体观测
                    - act_2/3: 其他智能体动作
            state (Tensor): RNN的隐藏状态，形状为 [B, H]
            seq_lens (Tensor): 序列长度，形状为 [B]

        Returns:
            output (Tensor): 输出张量，形状为 [B, num_outputs]
            new_state (Tensor): 更新后的隐藏状态，形状同输入state
        """
        
        # 本体观测数据拆分处理
        # obs_1_own被拆分为4个部分：
        # 前4列 | 中间N_OPP_HL个对手观测块 | 剩余字段
        self._inp1 = input_dict["obs"]["obs_1_own"][:,:4]
        self._inp2 = input_dict["obs"]["obs_1_own"][:,4:4+N_OPP_HL*OBS_OPP]
        self._inp3 = input_dict["obs"]["obs_1_own"][:,4+N_OPP_HL*OBS_OPP:]
        self._inp4 = input_dict["obs"]["obs_1_own"]

        # 多智能体观测-动作拼接
        # 每个智能体的观测与其动作拼接，最终合并所有智能体数据
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"]),dim=1)
        self._v2 = torch.cat((input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"]),dim=1)
        self._v3 = torch.cat((input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)
        self._v4 = torch.cat((self._v1, self._v2, self._v3), dim=1)

        # RNN核心处理
        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        # 输出形状调整
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, input_dict, state, seq_lens):
        """执行循环神经网络前向计算，处理输入并返回输出及更新后的隐藏状态
        
        Args:
            input_dict (dict): 输入特征字典，包含网络所需输入特征
            state (list): 包含两个元素的初始隐藏状态列表，对应动作网络和值函数网络
            seq_lens (Tensor): 序列长度张量，用于处理变长序列
            
        Returns:
            Tensor: 动作网络的输出特征
            list: 更新后的隐藏状态列表，包含动作网络和值函数网络的隐藏状态"""
        
        # 动作网络分支处理流程
        # 合并三个基础输入分支的特征
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2), self.inp3(self._inp3)),dim=1) 
        # 处理第四个输入分支并执行时序处理
        x_full = self.inp4(self._inp4)
        # 执行RNN时序特征提取（添加时间维度->RNN处理->移除时间维度）
        y, h = self.rnn_act(add_time_dimension(x_full, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[0], 0))
        # 融合时序特征与原始特征并进行归一化
        x_full = nn.functional.normalize(x_full + y.reshape(-1, 200))
        # 合并所有分支特征并进行后续处理
        x = torch.cat((x, x_full), dim=1)
        x = self.shared_layer(x)
        x = self.act_out(x)

        # 值函数网络分支处理流程（结构类似动作网络）
        z = torch.cat((self.v1(self._v1), self.v2(self._v2), self.v3(self._v3)),dim=1)
        z_full = self.v4(self._v4)
        w, k = self.rnn_val(add_time_dimension(z_full, seq_lens=seq_lens, framework="torch", time_major=False), torch.unsqueeze(state[1], 0))
        z_full = nn.functional.normalize(z_full + w.reshape(-1, 200))
        z = torch.cat((z, z_full), dim=1)
        z = self.shared_layer(z)
        self._val = self.val_out(z)

        # 返回处理结果并调整隐藏状态维度
        return x, [torch.squeeze(h,0), torch.squeeze(k, 0)]
    
    @override(ModelV2)
    def value_function(self):
        """
        重写父类方法，获取当前模型的状态价值估计
        
        返回:
            torch.Tensor: 重塑为一维张量的状态价值预测结果，
                        形状为[-1]表示自动推断维度大小
        
        Raises:
            AssertionError: 若未提前执行forward前向计算则抛出异常
        """
        # 前置校验：必须完成前向传播计算后才能获取价值
        assert self._val is not None, "must call forward first!"
        # 将多维价值张量展平为标准输出格式
        return torch.reshape(self._val, [-1])
