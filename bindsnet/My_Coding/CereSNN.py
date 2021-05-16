import torch
import matplotlib.pyplot as plt

from bindsnet.encoding.encodings import bernoulli_RBF, poisson_IO, IO_Current2spikes, Decode_Output
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, LIF_Train
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor, Global_Monitor,Our_Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_weights
from bindsnet.learning import STDP, IO_Record, PostPre, NoOp
from bindsnet.utils import Error2IO_Current
from bindsnet.encoding import poisson, bernoulli
from bindsnet.pipeline.environment_pipeline import MusclePipeline, TrajectoryPlanner
from bindsnet.environment.environment import MuscleEnvironment

# time = 50
# network
network = Network(dt=1)

# nodes
GR_Joint_layer = Input(n=100, traces=True)
PK = LIF_Train(n=32, traces=True,refrac=0,thresh=-40)
PK_Anti = LIF_Train(n=32, traces=True,refrac=0,thresh=-40)
IO = Input(n=32,traces=True)
IO_Anti = Input(n=32,traces=True)
DCN = LIFNodes(n=100, thresh=-57, traces=True)
DCN_Anti = LIFNodes(n=100, thresh=-57, trace=True)

# add Connection
Parallelfiber = Connection(
    source=GR_Joint_layer,
    target=PK,
    wmin=0,
    wmax=10,
    update_rule=STDP,
    nu=[0.1,0.3],
    w=0.1 + torch.zeros(GR_Joint_layer.n, PK.n),
)

Parallelfiber_Anti = Connection(
    source=GR_Joint_layer,
    target=PK_Anti,
    wmin=0,
    wmax=10,
    nu=[0.1,0.3],
    update_rule=STDP,
    w=0.1 + torch.zeros(GR_Joint_layer.n, PK_Anti.n)
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
    w=-0.1 * torch.ones(PK.n, DCN.n)
)

PK_DCN_Anti = Connection(
    source=PK_Anti,
    target=DCN_Anti,
    w=-0.1 * torch.ones(PK_Anti.n, DCN_Anti.n)
)

GR_DCN = Connection(
    source=GR_Joint_layer,
    target=DCN,
    w=0.1 * torch.ones(GR_Joint_layer.n, DCN.n)
)

GR_DCN_Anti = Connection(
    source=GR_Joint_layer,
    target=DCN_Anti,
    w=0.1 * torch.ones(GR_Joint_layer.n, DCN_Anti.n)
)

network.add_layer(layer=GR_Joint_layer, name="GR_Joint_layer")
network.add_layer(layer=PK, name="PK")
network.add_layer(layer=PK_Anti, name="PK_Anti")
network.add_layer(layer=IO, name="IO")
network.add_layer(layer=IO_Anti, name="IO_Anti")
network.add_layer(layer=DCN, name="DCN")
network.add_layer(layer=DCN_Anti, name="DCN_Anti")
network.add_connection(connection=Climbingfiber, source="IO", target="PK")
network.add_connection(connection=Climbingfiber_Anti, source="IO_Anti", target="PK_Anti")
network.add_connection(connection=Parallelfiber, source="GR_Joint_layer", target="PK")
network.add_connection(connection=Parallelfiber_Anti, source="GR_Joint_layer", target="PK_Anti")

network.add_connection(connection=PK_DCN, source="PK", target="DCN")
network.add_connection(connection=PK_DCN_Anti, source="PK_Anti", target="DCN_Anti")

network.add_connection(connection=GR_DCN, source="GR_Joint_layer", target="DCN")
network.add_connection(connection=GR_DCN_Anti, source="GR_Joint_layer", target="DCN_Anti")

GR_monitor = Monitor(
    obj=GR_Joint_layer,
    state_vars=("s"),

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
    obj=IO,
    state_vars=("s"),

)
IO_Anti_monitor = Monitor(
    obj=IO_Anti,
    state_vars=("s"),

)

DCN_monitor = Monitor(
    obj=DCN,
    state_vars=("s", "v"),

)

DCN_Anti_monitor = Monitor(
    obj=DCN_Anti,
    state_vars=("s", "v"),

)
network.add_monitor(monitor=GR_monitor, name="GR")
network.add_monitor(monitor=PK_monitor, name="PK")
network.add_monitor(monitor=PK_Anti_monitor, name="PK_Anti")
network.add_monitor(monitor=IO_monitor, name="IO")
network.add_monitor(monitor=IO_Anti_monitor, name="IO_Anti")
network.add_monitor(monitor=DCN_monitor, name="DCN")
network.add_monitor(monitor=DCN_Anti_monitor, name="DCN_Anti")

T = TrajectoryPlanner()
T.generate()

env = MuscleEnvironment()
My_pipe = MusclePipeline(network=network,
                         environment=env,
                         planner=T,
                         encoding_time=50,
                         total_time=5000,
                         receive_list=["network", "anti_network"],
                         send_list=["pos", "vel"],
                         kv=1,
                         kx=10)


def run_pipeline(MusclePipeline, episode_count):
    for i in range(episode_count):
        MusclePipeline.reset_state_variables()
        while not MusclePipeline.is_done:
            MusclePipeline.step()
        # spikes = {"PK":MusclePipeline.our_monitor.get("s")}
        # plt.ioff()
        #
        # plot_spikes(spikes)
    MusclePipeline.env.close()


print("-"*10+"Training"+"-"*10)
run_pipeline(My_pipe,1)
My_pipe.network.save("net")
plt.show()
print("-"*10+"Testing"+"-"*10)
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
