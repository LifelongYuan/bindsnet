import itertools
from typing import Callable, Optional, Tuple, Dict
from math import sin, pi
import torch
import time
from tqdm import tqdm
import numpy as np
from .base_pipeline import BasePipeline
from ..analysis.pipeline_analysis import MatplotlibAnalyzer
from ..environment import Environment, MuscleEnvironment
from ..network import Network
from ..network.nodes import AbstractInput
from ..network.monitors import Monitor, Global_Monitor, Our_Monitor
from bindsnet.encoding import bernoulli_RBF, bernoulli_pre, poisson_IO, IO_Current2spikes, Decode_Output, bernoulli
from bindsnet.utils import Error2IO_Current
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
import matplotlib.pyplot as plt
import math


class TrajectoryPlanner:
    def __init__(self, plan_time):
        self.plan_time = plan_time
        self.step_time = 0.1
        self.p = np.zeros((int(self.plan_time / self.step_time) + 1))
        self.v = np.zeros((int(self.plan_time / self.step_time) + 1))
        self.a = np.zeros((int(self.plan_time / self.step_time) + 1))

    def generate(self):
        # min_theta = 0
        # max_theta = 1
        # for i in range(0, int(self.plan_time / self.step_time + 1)):
        #     if i < int(self.plan_time / self.step_time/2):
        #         self.p[i] = min_theta
        #     else:
        #         self.p[i] = max_theta
        for i in range(0, int(self.plan_time / self.step_time + 1)):
            self.p[i] = 2.4 * sin(0.1 * 0.2 * i - pi / 2) + 2.4

    def pos_output(self, n_step) -> float:
        """
        Output
        """
        return self.p[n_step]

    def vel_output(self, n_step) -> float:
        return self.v[n_step]

    def acc_output(self, n_step) -> float:
        return self.v[n_step]


class EnvironmentPipeline(BasePipeline):
    # language=rst
    """
    Abstracts the interaction between ``Network``, ``Environment``, and environment
    feedback action.
    """

    def __init__(
            self,
            network: Network,
            environment: Environment,
            action_function: Optional[Callable] = None,
            encoding: Optional[Callable] = None,
            **kwargs,
    ):
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        :param environment: Arbitrary environment.
        :param action_function: Function to convert network outputs into environment inputs.
        :param encoding: Function to encoding input.

        Keyword arguments:

        :param str device: PyTorch computing device
        :param encode_factor: coefficient for the input before encoding.
        :param int num_episodes: Number of episodes to train for. Defaults to 100.
        :param str output: String name of the layer from which to take output.
        :param int render_interval: Interval to render the environment.
        :param int reward_delay: How many iterations to delay delivery of reward.
        :param int time: Time for which to run the network. Defaults to the network's
        :param int overlay_input: Overlay the last X previous input
        :param float percent_of_random_action: chance to choose random action
        :param int random_action_after: take random action if same output action counter reach

            timestep.
        """
        super().__init__(network, **kwargs)

        self.episode = 0

        self.env = environment
        self.action_function = action_function
        self.encoding = encoding

        self.accumulated_reward = 0.0
        self.reward_list = []

        # Setting kwargs.
        self.num_episodes = kwargs.get("num_episodes", 100)
        self.output = kwargs.get("output", None)
        self.render_interval = kwargs.get("render_interval", None)
        self.plot_interval = kwargs.get("plot_interval", None)
        self.reward_delay = kwargs.get("reward_delay", None)
        self.time = kwargs.get("time", int(network.dt))
        self.overlay_t = kwargs.get("overlay_input", 1)
        self.percent_of_random_action = kwargs.get("percent_of_random_action", 0.0)
        self.encode_factor = kwargs.get("encode_factor", 1.0)

        if torch.cuda.is_available() and self.allow_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # var for overlay process
        if self.overlay_t > 1:
            self.overlay_time_effect = torch.tensor(
                [i / self.overlay_t for i in range(1, self.overlay_t + 1)],
                dtype=torch.float,
                device=self.device,
            )
        self.overlay_start = True
        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)

        # Set up for multiple layers of input layers.
        self.inputs = [
            name
            for name, layer in network.layers.items()
            if isinstance(layer, AbstractInput)
        ]

        self.action = torch.tensor(-1, device=self.device)
        self.last_action = torch.tensor(-1, device=self.device)
        self.action_counter = 0
        self.random_action_after = kwargs.get("random_action_after", self.time)

        self.voltage_record = None
        self.threshold_value = None
        self.reward_plot = None
        self.first = True

        self.analyzer = MatplotlibAnalyzer(**self.plot_config)

        if self.output is not None:
            self.network.add_monitor(
                Monitor(self.network.layers[self.output], ["s"], time=self.time),
                self.output,
            )

            self.spike_record = {
                self.output: torch.zeros((self.time, self.env.action_space.n)).to(
                    self.device
                )
            }

    def init_fn(self) -> None:
        pass

    def train(self, **kwargs) -> None:
        # language=rst
        """
        Trains for the specified number of episodes. Each episode can be of arbitrary
        length.
        """
        while self.episode < self.num_episodes:
            self.reset_state_variables()

            for _ in itertools.count():
                obs, reward, done, info = self.env_step()

                self.step((obs, reward, done, info), **kwargs)

                if done:
                    break

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )
            self.episode += 1

    def env_step(self) -> Tuple[torch.Tensor, float, bool, Dict]:
        # language=rst
        """
        Single step of the environment which includes rendering, getting and performing
        the action, and accumulating/delaying rewards.

        :return: An OpenAI ``gym`` compatible tuple with modified reward and info.
        """
        # Render game.
        if (
                self.render_interval is not None
                and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            self.last_action = self.action
            if torch.rand(1) < self.percent_of_random_action:
                self.action = torch.randint(
                    low=0, high=self.env.action_space.n, size=(1,)
                )[0]
            elif self.action_counter > self.random_action_after:
                if self.last_action == 0:  # last action was start b
                    self.action = 1  # next action will be fire b
                    tqdm.write(f"Fire -> too many times {self.last_action} ")
                else:
                    self.action = torch.randint(
                        low=0, high=self.env.action_space.n, size=(1,)
                    )[0]
                    tqdm.write(f"too many times {self.last_action} ")
            else:
                self.action = self.action_function(self, output=self.output)

            if self.last_action == self.action:
                self.action_counter += 1
            else:
                self.action_counter = 0

        # Run a step of the environment.
        obs, reward, done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward

        info["accumulated_reward"] = self.accumulated_reward

        return obs, reward, done, info

    def step_(
            self, gym_batch: Tuple[torch.Tensor, float, bool, Dict], **kwargs
    ) -> None:
        # language=rst
        """
        Run a single iteration of the network and update it and the reward list when
        done.

        :param gym_batch: An OpenAI ``gym`` compatible tuple.
        """
        obs, reward, done, info = gym_batch

        if self.overlay_t > 1:
            if self.overlay_start:
                self.overlay_last_obs = (
                    obs.view(obs.shape[2], obs.shape[3]).clone().to(self.device)
                )
                self.overlay_buffer = torch.stack(
                    [self.overlay_last_obs] * self.overlay_t, dim=2
                ).to(self.device)
                self.overlay_start = False
            else:
                obs = obs.to(self.device)
                self.overlay_next_stat = torch.clamp(
                    self.overlay_last_obs - obs, min=0
                ).to(self.device)
                self.overlay_last_obs = obs.clone()
                self.overlay_buffer = torch.cat(
                    (
                        self.overlay_buffer[:, :, 1:],
                        self.overlay_next_stat.view(
                            [
                                self.overlay_next_stat.shape[2],
                                self.overlay_next_stat.shape[3],
                                1,
                            ]
                        ),
                    ),
                    dim=2,
                )
            obs = (
                    torch.sum(self.overlay_time_effect * self.overlay_buffer, dim=2)
                    * self.encode_factor
            )

        # Place the observations into the inputs.
        if self.encoding is None:
            obs = obs.unsqueeze(0).unsqueeze(0)
            obs_shape = torch.tensor([1] * len(obs.shape[1:]), device=self.device)
            inputs = {
                k: self.encoding(
                    obs.repeat(self.time, *obs_shape).to(self.device),
                    device=self.device,
                )
                for k in self.inputs
            }
        else:
            obs = obs.unsqueeze(0)
            inputs = {
                k: self.encoding(obs, self.time, device=self.device)
                for k in self.inputs
            }
            print(inputs)

        # Run the network on the spike train-encoded inputs.
        self.network.run(inputs=inputs, time=self.time, reward=reward, **kwargs)

        if self.output is not None:
            self.spike_record[self.output] = (
                self.network.monitors[self.output].get("s").float()
            )

        if done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward=self.accumulated_reward,
                    steps=self.step_count,
                    **kwargs,
                )
            self.reward_list.append(self.accumulated_reward)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_state_variables()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.overlay_start = True
        self.action = torch.tensor(-1)
        self.last_action = torch.tensor(-1)
        self.action_counter = 0

    def plots(self, gym_batch: Tuple[torch.Tensor, float, bool, Dict], *args) -> None:
        # language=rst
        """
        Plot the encoded input, layer spikes, and layer voltages.

        :param gym_batch: An OpenAI ``gym`` compatible tuple.
        """
        if self.plot_interval is None:
            return

        obs, reward, done, info = gym_batch

        for key, item in self.plot_config.items():
            if key == "obs_step" and item is not None:
                if self.step_count % item == 0:
                    self.analyzer.plot_obs(obs[0, ...].sum(0))
            elif key == "data_step" and item is not None:
                if self.step_count % item == 0:
                    self.analyzer.plot_spikes(self.get_spike_data())
                    self.analyzer.plot_voltages(*self.get_voltage_data())
            elif key == "reward_eps" and item is not None:
                if self.episode % item == 0 and done:
                    self.analyzer.plot_reward(self.reward_list)

        self.analyzer.finalize_step()


class MusclePipeline(BasePipeline):
    # language=rst
    """
    Abstracts the interaction between ``Network``, ``Environment``, and environment
    feedback action.
    """

    def __init__(
            self,
            network: Network,
            environment: MuscleEnvironment,
            planner: TrajectoryPlanner,
            encoding_time: int,
            total_time: float,
            send_list: list,
            receive_list: list,
            kv: float,
            kx: float,
            error_max: float = 8,
            out_max: float = 3,
            **kwargs,
    ):
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        :param environment: Arbitrary environment.
        :param planner: Trajectory planner
        :param encoding_time: encoding time for one input
        :param total_time: total running time
        :param send_list: send from env to network
        :param receive_list: send from network to env
        :param kv: error co for vel
        :param kx: error co for pos
        """

        super().__init__(network, **kwargs)

        self.episode = 0
        self.plot_interval = kwargs.get("plot_interval", None)
        if self.plot_config["data_length"] is not encoding_time:
            print("plot_time mismatch")
            self.plot_config["data_length"] = encoding_time
        self.analyzer = MatplotlibAnalyzer(**self.plot_config)

        # save four mainly use obj
        self.env = environment
        self.Info_network = {"pos": 0.0, "vel": 0.0, "network": 0.0, "anti_network": 0.0}
        self.planner = planner
        self.global_monitor = Global_Monitor(muscle_vars=["pos", "vel"],
                                             net_vars=["network", "anti_network"],
                                             )
        self.our_monitor = Our_Monitor(
            obj=self.network.layers["PK"],
            state_vars=("s", "v")
        )

        # para related with time scale
        self.encoding_time = encoding_time
        self.total_time = total_time
        self.step_now = 0  # record which step the pipeline is in

        self.send_list = send_list
        self.receive_list = receive_list

        self.kv = kv
        self.kx = kx
        self.error_max = error_max
        self.out_max = out_max
        self.is_done = False
        # set GPU or CPU
        if torch.cuda.is_available() and self.allow_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # generate trajectory
        self.planner.generate()
        self.env.start(sim_name=kwargs.get("sim_name", "actuator_2"))
        self.REC_DICT = {"error": 0.0, "curr": 0.0, "curr_anti": 0.0}
        self.REC = {"Pressure": [], "Anti_Pressure": [], "input": [], "error": [], "tout": []}  # record

    def init_fn(self) -> None:
        pass

    def step_(self, batch=1, **kwargs) -> int:
        # language=rst
        """
        Single step of the environment which includes rendering, getting and performing
        the action, and accumulating/delaying rewards.

        """
        # encode desired joint position
        # TODO only pos no vel
        Input_RATE = bernoulli_pre(datum=self.planner.pos_output(self.step_now), num_group=1)
        desired_pos = bernoulli_RBF(datum=Input_RATE,
                                    neural_num=self.network.layers["MF_layer"].n,
                                    time=self.encoding_time,
                                    dt=self.network.dt,
                                    num_group=1
                                    )

        # desired_vel =self.encoding(self.planner.vel_output(self.step_now),
        #                             self.network.layers["GR_Joint_layer"].n,
        #                             self.encoding_time,
        #                             self.network.dt
        #                             )

        self.Sender()
        error = self.kx * (self.planner.pos_output(self.step_now) - self.Info_network["pos"])

        curr, curr_anti = Error2IO_Current(datum=error, error_max=self.error_max)
        self.REC_DICT["error"] = error
        self.REC_DICT["curr"] = curr
        self.REC_DICT["curr_anti"] = curr_anti

        IO_input = IO_Current2spikes(curr,
                                     neural_num=self.network.layers["IO"].n,
                                     time=self.encoding_time,
                                     dt=self.network.dt,
                                     max_prob=0.9,

                                     )
        IO_anti_input = IO_Current2spikes(curr_anti,
                                          neural_num=self.network.layers["IO"].n,
                                          time=self.encoding_time,
                                          dt=self.network.dt,
                                          max_prob=0.9
                                          )
        inputs = {
            "IO": IO_input,
            "MF_layer": desired_pos,
            "IO_Anti": IO_anti_input,
            # "IO_new": IO_input,
            # "IO_Anti_new": IO_anti_input
        }
        # run the network and write into the Info_network
        self.network_run(inputs)
        # send from info_net to info_err
        self.Receiver()
        # eng step
        self.env.step(real_time=self.step_now * 0.1,
                      record_list=['pos', 'vel'],
                      command_list=['network', 'anti_network'])
        # monitor add
        self.global_monitor.record(self.env.Info_muscle, self.Info_network)
        self.our_monitor.record()
        # step sign ++
        self.step_now += 1
        if self.step_now >= (self.total_time / self.encoding_time):
            self.is_done = True
        self.record_data()
        return 1

    def Sender(self):
        for l in self.send_list:
            assert self.env.Info_muscle.get(l) is not None, "No such key in source list"
            self.Info_network[l] = self.env.Info_muscle[l]

    def Receiver(self):
        for l in self.receive_list:
            assert self.Info_network.get(l) is not None, "No such key in source list"
            self.env.Info_muscle[l] = self.Info_network[l]

    def network_run(self, inputs: Dict):
        self.network.run(inputs=inputs, time=self.encoding_time)
        # DCN = self.network.monitors["DCN"].get("s")
        # plot_spikes({"DCN":DCN})
        # Output = Decode_Output(DCN, self.network.layers["DCN"].n, self.encoding_time, self.network.dt, 10.0)
        # DCN_Anti = self.network.monitors["DCN_Anti"].get("s")
        # Output_Anti = Decode_Output(DCN_Anti, self.network.layers["DCN_Anti"].n, self.encoding_time, self.network.dt, 10.0)
        if self.step_now is 0:
            self.Info_network["network"] = 0
            self.Info_network["anti_network"] = 0
            DCN = self.network.monitors["DCN"].get("s")
            DCN_Anti = self.network.monitors["DCN_Anti"].get("s")

        else:
            DCN = self.network.monitors["DCN"].get("s")
            Output = Decode_Output(DCN, self.network.layers["DCN"].n, self.encoding_time, self.network.dt,
                                   bound_width=self.out_max)
            DCN_Anti = self.network.monitors["DCN_Anti"].get("s")
            Output_Anti = Decode_Output(DCN_Anti, self.network.layers["DCN_Anti"].n, self.encoding_time,
                                        self.network.dt, bound_width=self.out_max)
            # PK = self.network.monitors["PK"].get("s")
            # Output = Decode_Output(PK, self.network.layers["PK"].n, self.encoding_time, self.network.dt, 1)
            # PK_Anti = self.network.monitors["PK_Anti"].get("s")
            # Output_Anti = Decode_Output(PK_Anti, self.network.layers["PK_Anti"].n, self.encoding_time, self.network.dt,
            #                             1)
            self.Info_network["network"] = float(Output)
            self.Info_network["anti_network"] = float(Output_Anti)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        if self.step_now is not 0:
            self.env.close()
        self.env.reset()
        self.network.reset_state_variables()
        self.step_now = 0
        self.is_done = False
        super().reset_state_variables()

    def plots(self, batch=1, *args) -> None:
        # language=rst
        """
        Plot the encoded input, layer spikes, and layer voltages.

        :param batch: default 1
        """

        if self.plot_interval is None:
            return

        for key, item in self.plot_config.items():
            # if key == "obs_step" and item is not None:
            #     if self.step_count % item == 0:
            #         self.analyzer.plot_obs(obs[0, ...].sum(0))
            if key == "data_step" and item is not None:
                if self.step_count % item == 0:
                    self.analyzer.plot_spikes(self.get_spike_data())
                    self.analyzer.plot_voltages(*self.get_voltage_data())
                    self.analyzer.plot_reward(show_list=self.REC, tag="Values")

            # elif key == "reward_eps" and item is not None:
            #     if self.episode % item == 0 and done:
            #         self.analyzer.plot_reward(self.reward_list)

        self.analyzer.finalize_step()

    def print_message(self) -> None:
        # language=rst
        """
        Plot the encoded input, layer spikes, and layer voltages.

        :param batch: default 1
        """
        print("-" * 10 + "Error and input" + "-" * 10)
        print("error: {} = Kx * desire_pos: {} - real_pos: {}".format(self.REC_DICT["error"],
                                                                      self.planner.pos_output(self.step_now - 1),
                                                                      self.Info_network["pos"]))
        print("Error to Current: curr: {}   curr_anti:{}".format(self.REC_DICT["curr"], self.REC_DICT["curr_anti"]))
        print("Curr to spikes")
        print("-" * 10 + "Net running" + "-" * 10)
        # print("Weight:{}".format(self.network.connections[("GR_Joint_layer", "PK")].w))
        print("weight:{}".format(self.network.connections[("GR_Joint_layer", "PK")].w))
        # print("weight:{}".format(self.network.connections[("GR_Joint_layer","PK_Anti")].w))
        print("Pressure_add: {}   Anti_Pressure_add: {}".format(self.Info_network["network"],
                                                                self.Info_network["anti_network"]))

        pass

    def record_data(self) -> None:
        # record data and finally plot
        self.REC["Pressure"].append(self.Info_network["network"])
        self.REC["Anti_Pressure"].append(self.Info_network["anti_network"])
        self.REC["input"].append(0.5 * self.planner.pos_output(self.step_now - 1))
        self.REC["error"].append(self.REC_DICT["error"])
        self.REC["tout"].append(self.env.t_now)
        pass

    def data_analysis(self) -> None:
        plt.ioff()
        plt.figure()
        for l in self.REC:
            if l is "tout":
                continue
            plt.plot(self.REC["tout"], self.REC[l])
        plt.savefig('Values_{}.png'.format(self.epoch))
        self.REC = {"Pressure": [], "Anti_Pressure": [], "input": [], "error": [], "tout": []}  # record
        self.env.eng.save("valuess_{}".format(self.epoch), "valuess", nargout=0)
        self.env.eng.save("tout_{}".format(self.epoch), "tout", nargout=0)
        self.env.eng.save("pos_{}".format(self.epoch), "pos", nargout=0)
        self.env.eng.save("simlog_{}".format(self.epoch), "simlog", nargout=0)
