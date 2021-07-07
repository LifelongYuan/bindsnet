import torch
import matplotlib.pyplot as plt

from bindsnet.encoding.encodings import bernoulli_RBF, poisson_IO, IO_Current2spikes, Decode_Output
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, LIF_Train
from bindsnet.network.topology import Connection, Group_Connection,SparseConnection
from bindsnet.network.monitors import Monitor, Global_Monitor, Our_Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
from bindsnet.learning import STDP, IO_Record, PostPre, NoOp
from bindsnet.utils import Error2IO_Current
from bindsnet.encoding import poisson, bernoulli
from bindsnet.pipeline.environment_pipeline import MusclePipeline, TrajectoryPlanner
from bindsnet.environment.environment import MuscleEnvironment
import matlab
# time = 50
# network
network = Network(dt=1)

# nodes
MF_layer = Input(n=80, traces=True, refrac=0)  # 50 group each 8  10 g 8
GR_Joint_layer = LIFNodes(n=100, traces=True, refrac=0)
PK = LIF_Train(n=80, traces=True, refrac=0)
PK_Anti = LIF_Train(n=80, traces=True, refrac=0)
IO = Input(n=80, traces=True, is_IO=True, refrac=0)
IO_Anti = Input(n=80, traces=True, is_IO=True, refrac=0)
IO_new = Input(n=80, traces=True, refrac=0)
IO_Anti_new = Input(n=80, traces=True, refrac=0)
DCN = LIFNodes(n=80, traces=True, refrac=0)
DCN_Anti = LIFNodes(n=80, trace=True, refrac=0)

MF_fiber = SparseConnection(
    source=MF_layer,
    target=GR_Joint_layer,
    sparsity = 0.5

)

# add Connection
Parallelfiber = Connection(
    source=GR_Joint_layer,
    target=PK,
    wmin=0.1,
    wmax=5,
    update_rule=STDP,
    nu=[0.01, 0.05],
    w=0.1 + torch.zeros(GR_Joint_layer.n, PK.n),
    # norm=0.3 * GR_Joint_layer.n
)
Parallelfiber_Anti = Connection(
    source=GR_Joint_layer,
    target=PK_Anti,
    wmin=0.1,
    wmax=5,
    nu=[0.01, 0.05],
    update_rule=STDP,
    w=0.1 + torch.zeros(GR_Joint_layer.n, PK_Anti.n),
    # norm=0.3 * GR_Joint_layer.n
)

Climbingfiber = Connection(
    source=IO,
    target=PK,
    update_rule=IO_Record,

)

Climbingfiber_Anti = Connection(
    source=IO_Anti,
    target=PK_Anti,
    update_rule=IO_Record,


)

PK_DCN = Connection(
    source=PK,
    target=DCN,
    # w=-0.1 * torch.ones(PK.n, DCN.n)
    norm=-0.1*PK.n
)

PK_DCN_Anti = Connection(
    source=PK_Anti,
    target=DCN_Anti,
    # w=-0.1 * torch.ones(PK_Anti.n, DCN_Anti.n)
    norm=-0.1*PK.n
)

IO_DCN = Connection(
    source=GR_Joint_layer,
    target=DCN,
    # w=0.1 * torch.ones(IO_new.n, DCN.n)
    norm=0.5*GR_Joint_layer.n

)

IO_DCN_Anti = Connection(
    source=GR_Joint_layer,
    target=DCN_Anti,
    # w=0.1 * torch.ones(IO_Anti_new.n, DCN_Anti.n)
    norm=0.5*GR_Joint_layer.n

)

network.add_layer(layer=MF_layer, name="MF_layer")
network.add_layer(layer=GR_Joint_layer, name="GR_Joint_layer")
network.add_layer(layer=PK, name="PK")
network.add_layer(layer=PK_Anti, name="PK_Anti")
network.add_layer(layer=IO, name="IO")
network.add_layer(layer=IO_Anti, name="IO_Anti")
network.add_layer(layer=DCN, name="DCN")
network.add_layer(layer=DCN_Anti, name="DCN_Anti")
network.add_layer(layer=IO_new, name="IO_new")
network.add_layer(layer=IO_Anti_new, name="IO_Anti_new")

network.add_connection(connection=MF_fiber, source="MF_layer", target="GR_Joint_layer")
network.add_connection(connection=Climbingfiber, source="IO", target="PK")
network.add_connection(connection=Climbingfiber_Anti, source="IO_Anti", target="PK_Anti")
network.add_connection(connection=Parallelfiber, source="GR_Joint_layer", target="PK")
network.add_connection(connection=Parallelfiber_Anti, source="GR_Joint_layer", target="PK_Anti")

network.add_connection(connection=PK_DCN, source="PK", target="DCN")
network.add_connection(connection=PK_DCN_Anti, source="PK_Anti", target="DCN_Anti")
network.add_connection(connection=IO_DCN, source="GR_Joint_layer", target="DCN")
network.add_connection(connection=IO_DCN_Anti, source="GR_Joint_layer", target="DCN_Anti")

MF_monitor = Monitor(
    obj=MF_layer,
    state_vars="s"
)

GR_monitor = Monitor(
    obj=GR_Joint_layer,
    state_vars=("s", "v"),

)
PK_monitor = Monitor(
    obj=PK,
    state_vars=("s", "v")
)

PK_Anti_monitor = Monitor(
    obj=PK_Anti,
    state_vars=("s", "v"),

)
IO_monitor = Monitor(
    obj=IO_new,
    state_vars=("s")
)
IO_Anti_monitor = Monitor(
    obj=IO_Anti_new,
    state_vars=("s")
)

DCN_monitor = Monitor(
    obj=DCN,
    state_vars=("s", "v"),

)

DCN_Anti_monitor = Monitor(
    obj=DCN_Anti,
    state_vars=("s", "v"),

)
network.add_monitor(monitor=MF_monitor, name="MF")
network.add_monitor(monitor=GR_monitor, name="GR")
network.add_monitor(monitor=PK_monitor, name="PK")
network.add_monitor(monitor=PK_Anti_monitor, name="PK_Anti")
network.add_monitor(monitor=IO_monitor, name="IO_monitor")
network.add_monitor(monitor=IO_Anti_monitor, name="IO_Anti_monitor")
network.add_monitor(monitor=DCN_monitor, name="DCN")
network.add_monitor(monitor=DCN_Anti_monitor, name="DCN_Anti")
T = TrajectoryPlanner(plan_time=200)
T.generate()
env = MuscleEnvironment()
My_pipe = MusclePipeline(network=network,
                         environment=env,
                         save_dir="PTH",
                         sim_name='actuator_2',
                         save_interval=1,
                         print_interval=1,
                         plot_interval=None,
                         plot_config={"data_step": True, "data_length": 50, "volts_type": "line"},
                         planner=T,
                         encoding_time=50,
                         total_time=25000,
                         receive_list=["network", "anti_network"],
                         send_list=["pos", "vel"],
                         allow_gpu=False,
                         kv=1,
                         kx=1,
                         error_max=0.5,
                         out_max=4.5,
                         )



def run_pipeline(pipeline, episode_count):
    for i in range(episode_count):
        pipeline.reset_state_variables()
        while not pipeline.is_done:
            pipeline.step(1)
        # spikes = {"PK":MusclePipeline.our_monitor.get("s")}
        # plt.ioff()
        #
        # plot_spikes(spikes)
        pipeline.data_analysis()    # final plot
    # pipeline.env.close()


print("-" * 10 + "Training" + "-" * 10)
run_pipeline(My_pipe, 4)
print("-" * 10 + "Testing" + "-" * 10)
#
# spikes = {
#     "GR": My_pipe.network.monitors["GR"].get("s"),
#     "PK": My_pipe.network.monitors["PK"].get("s"),
#     #  "PK_Anti":PK_Anti_monitor.get("s"),
#     "IO": My_pipe.network.monitors["IO"].get("s")
#     # "DCN_Anti":DCN_Anti_monitor.get("s")
# }
# # spikes2 = {
# #     # "GR": GR_monitor.get("v"),
# #     "PK": PK_monitor.get("s")
# #     #  "PK_Anti":PK_Anti_monitor.get("s"),
# #     # "IO":IO_monitor.get("s"),
# #     #  "DCN":DCN_monitor.get("s"),
# #     # "DCN_Anti":DCN_Anti_monitor.get("s")
# # }
# #
# # weight = Parallelfiber.w
# # plot_weights(weights=weight)
# # voltages = {
# #     "DCN": DCN_monitor.get("v"),
# #     "PK": PK_monitor.get("v"),
# #     "PK_Anti": PK_Anti_monitor.get("v")
# # }
# plt.ioff()
# plot_spikes(spikes)
# # print("---- Output of DCN neural ----")
# # DCN = DCN_monitor.get("s")
# # DCN_Anti = DCN_Anti_monitor.get("s")
# # plot_voltages(voltages, plot_type="line")
# # plt.show()
