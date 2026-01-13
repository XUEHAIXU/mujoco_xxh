import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R
import os

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")


def write_flag_file(flag_filename="rl_visu_flag"):
    """创建可视化标志文件，防止多进程重复启动可视化窗口"""
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        print(f"创建标志文件失败: {e}")
        return False


def check_flag_file(flag_filename="rl_visu_flag"):
    """检查可视化标志文件是否存在"""
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)


def delete_flag_file(flag_filename="rl_visu_flag"):
    """删除可视化标志文件"""
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        print(f"删除标志文件失败: {e}")
        return False


class PandaObstacleEnv(gym.Env):
    """
    熊猫机械臂目标到达环境
    核心功能：控制机械臂末端到达随机生成的目标位置，带有直线运动奖励、碰撞惩罚等
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, visualize: bool = False):
        super(PandaObstacleEnv, self).__init__()
        # 可视化控制：通过标志文件防止多进程重复启动可视化窗口
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        
        # Mujoco相关初始化
        self.handle = None  # 可视化窗口句柄
        self.model_path = './model/franka_emika_panda/scene.xml'  # 请根据实际路径修改
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化可视化窗口
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            # 设置相机参数
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        # 末端执行器ID（需与scene.xml中的body名称对应）
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        if self.end_effector_id == -1:
            raise ValueError("未找到末端执行器body: ee_center_body，请检查scene.xml")
        
        # 机械臂初始关节位姿（home位姿）
        self.home_joint_pos = np.array([
            0.0, -np.pi/4, 0.0, -3*np.pi/4,
            0.0, np.pi/2, np.pi/4
        ], dtype=np.float32)
        self.initial_ee_pos = np.zeros(3, dtype=np.float32)
        self.start_ee_pos = np.zeros(3, dtype=np.float32)
        
        # 目标点参数
        self.goal_size = 0.03
        self.goal_threshold = 0.005  # 到达目标的距离阈值
        self.goal = np.zeros(3, dtype=np.float32)
        
        # 工作空间约束（防止机械臂超出合理范围）
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.3]
        }
        
        # 动作空间：7个关节的归一化控制指令（-1~1）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # 观测空间：7个关节角度 + 3个目标位置
        self.obs_size = 7 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        
        # 随机数生成器
        self.np_random = np.random.default_rng(None)
        # 动作相关变量
        self.prev_action = np.zeros(7, dtype=np.float32)
        # 直线运动跟踪变量
        self.min_linearity_error = np.inf
        # 时间相关
        self.start_t = 0.0

    def _get_valid_goal(self) -> np.ndarray:
        """生成有效目标点：与初始位置距离在0.4~0.5之间，且x>0.2，z>0.2"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            dist = np.linalg.norm(goal - self.initial_ee_pos)
            if 0.4 < dist < 0.5 and goal[0] > 0.2 and goal[2] > 0.2:
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """在可视化窗口中渲染目标点（蓝色球体）"""
        if not self.visualize or self.handle is None:
            return
        
        # 清空之前的几何图形
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)  # 蓝色半透明
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 重置随机数生成器
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # 重置Mujoco数据
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        
        # 更新初始末端位置
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()
        
        # 生成新目标点
        self.goal = self._get_valid_goal()
        
        # 重置直线运动跟踪变量
        self.min_linearity_error = np.inf
        # 重置上一步动作
        self.prev_action = np.zeros(7, dtype=np.float32)
        # 重置开始时间
        self.start_t = time.time()
        
        # 渲染目标点
        if self.visualize:
            self._render_scene()        
        
        # 获取初始观测
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """获取观测：7个关节角度 + 3个目标位置"""
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[float, float, float]:
        """
        计算奖励函数
        返回：总奖励、到目标的距离、姿态误差
        """
        # 1. 距离目标的奖励（非线性）
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        if dist_to_goal < self.goal_threshold:
            distance_reward = 100.0
        elif dist_to_goal < 2 * self.goal_threshold:
            distance_reward = 50.0
        elif dist_to_goal < 3 * self.goal_threshold:
            distance_reward = 10.0
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)

        # 2. 直线运动奖励与远离惩罚
        start_to_goal = self.goal - self.start_ee_pos
        start_to_goal_norm = np.linalg.norm(start_to_goal)
        linearity_reward = 0.0
        deviation_penalty = 0.0
        
        if start_to_goal_norm >= 1e-6:
            start_to_current = ee_pos - self.start_ee_pos
            # 计算投影比例（限制在0~1之间）
            projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm ** 2)
            projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
            # 计算投影点和偏离距离
            projected_point = self.start_ee_pos + projection_ratio * start_to_goal
            linearity_error = np.linalg.norm(ee_pos - projected_point)
            
            # 直线接近奖励（离直线越近奖励越高）
            linearity_reward = 3.0 / (1.0 + linearity_error)
            # 远离趋势惩罚（比最近点更远时惩罚）
            if linearity_error < self.min_linearity_error:
                self.min_linearity_error = linearity_error
            else:
                deviation_penalty = 1.0 * (linearity_error - self.min_linearity_error)

        # 3. 姿态惩罚：保持末端朝下
        target_orient = np.array([0, 0, -1])
        ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
        dot_product = np.dot(ee_orient_norm, target_orient)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        orientation_penalty = 0.3 * angle_error
        
        # 4. 动作平滑性惩罚
        action_diff = action - self.prev_action
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        action_magnitude_penalty = 0.05 * np.linalg.norm(action)

        # 5. 碰撞惩罚
        contact_penalty = 1.0 * self.data.ncon
        
        # 6. 关节角度超限惩罚
        joint_penalty = 0.0
        for i in range(7):
            min_angle, max_angle = self.model.jnt_range[i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        # 7. 总奖励计算
        total_reward = (
            distance_reward
            + linearity_reward
            - contact_penalty
            - smooth_penalty
            - orientation_penalty
            - joint_penalty
            - deviation_penalty
            - action_magnitude_penalty
        )
        
        # 更新上一步动作
        self.prev_action = action.copy()
        
        return float(total_reward), float(dist_to_goal), float(angle_error)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        """
        执行一步环境交互
        返回：观测、奖励、是否终止、是否截断、信息字典
        """
        # 动作缩放：将[-1,1]映射到关节实际范围
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            min_j, max_j = joint_ranges[i]
            scaled_action[i] = min_j + (action[i] + 1) * 0.5 * (max_j - min_j)
        
        # 执行动作
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 获取末端状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(ee_quat)
        ee_orient = rot.as_euler('xyz')  # 欧拉角表示的姿态
        
        # 计算奖励
        reward, dist_to_goal, _ = self._calc_reward(ee_pos, ee_orient, self.data.qpos[:7], action)
        
        # 判断终止条件
        terminated = False
        truncated = False
        # 目标达成
        if dist_to_goal < self.goal_threshold:
            terminated = True
        # 超时（20秒）
        if time.time() - self.start_t > 20.0:
            reward -= 10.0
            terminated = True
        
        # 可视化更新
        if self.visualize and self.handle is not None:
            self._render_scene()
            self.handle.sync()
            time.sleep(1/self.metadata["render_fps"])  # 控制帧率
        
        # 获取新观测
        obs = self._get_observation()
        
        # 信息字典
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': self.data.ncon > 0
        }
        
        return obs, np.float32(reward), terminated, truncated, info

    def close(self) -> None:
        """关闭环境，释放资源"""
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:
    """
    训练PPO模型
    :param n_envs: 并行环境数
    :param total_timesteps: 总训练步数
    :param model_save_path: 模型保存路径
    :param visualize: 是否可视化（训练时建议关闭）
    :param resume_from: 从指定路径恢复训练
    """
    # 创建向量环境
    ENV_KWARGS = {'visualize': visualize}
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    # 加载或创建模型
    if resume_from is not None and os.path.exists(resume_from + ".zip"):
        print(f"从 {resume_from} 恢复训练")
        model = PPO.load(resume_from, env=env)
    else:
        print("创建新的PPO模型")
        POLICY_KWARGS = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 128], vf=[256, 128])]
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=2048,
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,
            learning_rate=2e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_reach_target/"
        )
    
    # 开始训练
    print(f"开始训练 | 并行环境数: {n_envs} | 总训练步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_ppo_reach_target",
    total_episodes: int = 5,
) -> None:
    """
    测试训练好的PPO模型
    :param model_path: 模型路径
    :param total_episodes: 测试轮数
    """
    # 创建测试环境（开启可视化）
    env = PandaObstacleEnv(visualize=True)
    
    # 加载模型
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"模型文件不存在: {model_path}.zip")
    model = PPO.load(model_path, env=env)
    
    # 测试过程
    success_count = 0
    print(f"\n开始测试 | 测试轮数: {total_episodes}")
    for ep in range(total_episodes):
        obs, _ = env.reset(seed=ep)  # 固定种子保证可复现
        done = False
        episode_reward = 0.0
        
        while not done:
            # 确定性预测（测试时使用）
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # 统计结果
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 距离目标: {info['distance_to_goal']:.4f} | 结果: {'成功' if info['is_success'] else '失败'}")
    
    # 输出统计信息
    success_rate = (success_count / total_episodes) * 100
    print(f"\n测试完成 | 总成功率: {success_rate:.1f}%")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 清理标志文件
    delete_flag_file()
    
    # 配置参数
    TRAIN_MODE = True  # True=训练，False=测试
    MODEL_PATH = "assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v6"
    # RESUME_MODEL_PATH = "assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v3"
    
    if TRAIN_MODE:
        # 训练模式
        train_ppo(
            n_envs=256,
            total_timesteps=500_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,  # 训练时建议关闭可视化
            # resume_from=RESUME_MODEL_PATH if os.path.exists(RESUME_MODEL_PATH + ".zip") else None
        )
    else:
        # 测试模式
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=15,
        )