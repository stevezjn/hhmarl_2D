import argparse
import os
import datetime

class Config(object):
    """
    Configurations for HHMARL Training. 
    Mode 0 = Low-level training
    Mode 1 = High-level training
    Mode 2 = Evaluation
    """
    def __init__(self, mode:int):
        self.mode = mode
        parser = argparse.ArgumentParser(description='HHMARL2D Training Config')

        # training mode
        parser.add_argument('--level', type=int, default=1, help='Training Level')
        parser.add_argument('--horizon', type=int, default=500, help='Length of horizon')
        parser.add_argument('--agent_mode', type=str, default="fight", help='Agent mode: Fight or Escape')
        parser.add_argument('--num_agents', type=int, default=2 if mode==0 else 3, help='Number of (trainable) agents')
        parser.add_argument('--num_opps', type=int, default=2 if mode==0 else 3, help='Number of opponents')
        parser.add_argument('--total_num', type=int, default=4 if mode==0 else 6, help='Total number of aircraft')
        parser.add_argument('--hier_opp_fight_ratio', type=int, default=75, help='Opponent fight policy selection probability [in %].')

        # env & training params
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore', type=bool, default=False, help='Restore from model')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--log_name', type=str, default=None, help='Experiment Name, defaults to Commander + date & time.')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel samplers')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=2000 if mode==0 else 1000, help='PPO train batch size')
        parser.add_argument('--mini_batch_size', type=int, default=256, help='PPO train mini batch size')
        parser.add_argument('--map_size', type=float, default=0.3 if mode==0 else 0.5, help='Map size -> *100 = [km]')

        # rewards
        parser.add_argument('--glob_frac', type=float, default=0, help='Fraction of reward sharing')
        parser.add_argument('--rew_scale', type=int, default=1, help='Reward scale')
        parser.add_argument('--esc_dist_rew', type=bool, default=False, help='Activate per-time-step reward for Escape Training.')
        parser.add_argument('--hier_action_assess', type=bool, default=True, help='Give action rewards to guide hierarchical training.')
        parser.add_argument('--friendly_kill', type=bool, default=True, help='Consider friendly kill or not.')
        parser.add_argument('--friendly_punish', type=bool, default=False, help='If friendly kill occurred, if both agents to punish.')

        # eval
        parser.add_argument('--eval_info', type=bool, default=True if mode==2 else False, help='Provide eval statistic in step() function or not. Dont change for evaluation.')
        parser.add_argument('--eval_hl', type=bool, default=True, help='True=evaluation with Commander, False=evaluation of low-level policies.')
        parser.add_argument('--eval_level_ag', type=int, default=5, help='Agent low-level for evaluation.')
        parser.add_argument('--eval_level_opp', type=int, default=4, help='Opponent low-level for evaluation.')
        
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()

    def set_metrics(self):
        """
        配置实验指标相关参数
        
        功能:
        1. 设置实验日志路径和名称
        2. 处理模型恢复逻辑
        3. 配置环境超参数
        4. 设置对抗等级和评估条件
        不包含返回值，所有配置存储于self.args对象中
        """

        # 配置日志路径和名称（根据模式选择不同命名规则）
        # mode=0时为常规训练模式，其他模式为指挥官模式
        #self.args.log_name = f'Commander_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        self.args.log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}' if self.mode == 0 else f'Commander_{self.args.num_agents}_vs_{self.args.num_opps}'
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        # 自动恢复策略检测（仅在首次训练时触发）
        # 战斗模式检查前一级别模型，逃生模式固定检查L3级模型
        if not self.args.restore and self.mode==0:
            if self.args.agent_mode == "fight" and os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level-1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')):
                self.args.restore = True
            elif self.args.agent_mode == "escape" and os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')):
                self.args.restore = True

        # 设置模型恢复路径（未指定路径时自动生成）
        # 战斗模式取前一级别模型，逃生模式固定取L3模型
        if self.args.restore:
            if self.args.restore_path is None:
                if self.mode == 0:
                    try:
                        if self.args.agent_mode=="fight":
                            # take previous pi_fight
                            self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level-1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}', 'checkpoint')
                        else:
                            # escape-vs-pi_fight
                            self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}', 'checkpoint')
                    except:
                        raise NameError(f'Could not restore previous {self.args.agent_mode} policy. Check restore_path.')
                else:
                    raise NameError('Specify full restore path to Commander Policy.')

        # 逃生模式特殊处理逻辑
        # 根据历史模型存在性调整训练等级（L3或L5）
        if self.args.agent_mode == "escape" and self.mode==0:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L3_escape_2-vs-2')):
                self.args.level = 3
            else:
                self.args.level = 5
            self.args.log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
            self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        # 设置环境时间步限制
        # 常规模式按等级设置不同步长，指挥官模式固定500步
        if self.mode == 0:
            horizon_level = {1: 150, 2:200, 3:300, 4:350, 5:400}
            self.args.horizon = horizon_level[self.args.level]
        else:
            self.args.horizon = 500

        # 指挥官模式评估设置
        # 强制对抗双方使用相同等级（L5）
        if self.mode == 2 and self.args.eval_hl:
            # when incorporating Commander, both teams are on same level.
            self.args.eval_level_ag = self.args.eval_level_opp = 5

        # 自动设置评估模式
        # 当启用渲染时强制进入评估状态
        self.args.eval = True if self.args.render else self.args.eval

        # 环境基础参数配置
        # 设置智能体总数和环境配置引用
        self.args.total_num = self.args.num_agents + self.args.num_opps
        self.args.env_config = {"args": self.args}

    @property
    def get_arguments(self):
        return self.args

