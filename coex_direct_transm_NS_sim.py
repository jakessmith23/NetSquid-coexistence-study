# All experimental values are taken from:
# Chapman, J. C., Lukens, J. M., Alshowkan, M., Rao, N., Kirby, B. T., & Peters, N. A. (2023). 
# Coexistent quantum channel characterization using spectrally resolved Bayesian quantum process tomography. 
# Physical Review Applied, 19(4), 044026.

import numpy as np
import matplotlib.pyplot as plt
import pandas
import netsquid as ns

from netsquid.protocols import NodeProtocol
from netsquid.protocols import Signals

from netsquid.nodes.connections import Connection
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components import QuantumChannel
import netsquid.qubits.ketstates as ks

from netsquid.components import QuantumChannel
from netsquid.components.models import DepolarNoiseModel
from netsquid.components.models import FibreDelayModel


from netsquid.util import DataCollector
from netsquid.protocols import Signals
import pydynaa

from netsquid.qubits.dmtools import DenseDMRepr

from netsquid.nodes import Node
from netsquid.nodes import Network
from netsquid.components import QuantumMemory
from netsquid.qubits.dmtools import DenseDMRepr

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EmitProtocol(NodeProtocol):
    def __init__(self, node, iterations, delay, verbose=False):
      # init parent NodeProtocol
      super().__init__(node)
      self.iterations = iterations
      self.delay = delay

      self.meas_results = []

    def run(self):
      for i in range(self.iterations):
        self.node.subcomponents['qsource'].trigger()
        yield self.await_timer(self.delay)

class ReceiveProtocol(NodeProtocol):
    def __init__(self, node, verbose=False):
      # init parent NodeProtocol
      super().__init__(node)

      self.verbose = verbose
      self.bp = None

    def run(self):
        if self.verbose: print({ns.sim_time()}, ": Starting", self.node.name, "s ReceiveProtocol")

        port_qin_emitter = self.node.ports["qin_emitter"]

        while True:

          yield self.await_port_input(port_qin_emitter)

          self.bp = None

          if self.verbose: print({ns.sim_time()}, self.node.name, "'s ReceiveProtocol received BP: ")
          bp, = port_qin_emitter.rx_input().items

          if self.verbose: print({ns.sim_time()}, self.node.name, "'s ReceiveProtocol peeking: ", bp)
          if self.verbose: print(ns.qubits.reduced_dm(bp))
          self.bp = bp

          self.send_signal(Signals.SUCCESS, False)


class QuantumConnection(Connection):
    def __init__(self, length, depolar_rate=0):
        # initialize the parent Connection
        super().__init__(name="QuantumConnection")

        models={"delay_model": FibreDelayModel(),
                "quantum_noise_model" : DepolarNoiseModel(depolar_rate=depolar_rate, time_independent=True),
                #'quantum_loss_model' : FibreLossModel(p_loss_length=attenuation_coeff)}
        }

        # add QuantumChannel subcomponent with associated models
        # forward A Port to ClassicalChannel send Port
        # forward ClassicalChannel recv Port to B Port
        self.add_subcomponent(QuantumChannel("qChannel_A2B", length=length,
                              models = models),
                              forward_input=[("A", "send")],
                              forward_output=[("B", "recv")])


def setup_datacollectors(prot_emitter, prot_rx):


    def get_fidelity(evexpr):
        raman_detected = prot_rx.get_signal_result(Signals.SUCCESS)

        b2 = prot_rx.bp

        return {
                "b2": b2,
                "dm": ns.qubits.reduced_dm([b2])}

    # init datacollector to call get_fidelity() when triggered
    dc_fidelity = DataCollector(get_fidelity, include_entity_name=False)
    # configure datacollector to trigger when Bob's Protocol signals SUCCESS
    dc_fidelity.collect_on(pydynaa.EventExpression(source=prot_rx,
                                          event_type=Signals.SUCCESS.value))

    return dc_fidelity


def get_char_noisy_state(s_x, s_y, s_z):
    
    # Define the Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Define the identity matrix
    I = np.eye(2)

    # Compute rho2 = 1/2 * (I + s_x * sigma_x + s_y * sigma_y + s_z * sigma_z)
    rho2 = 0.5 * (I + s_x * sigma_x + s_y * sigma_y + s_z * sigma_z)
    
    print("rho2:")
    print(rho2)
    return rho2



def characterized_network_setup(s_x, s_y, s_z, node_distance=4e-3, depolar_rate=0, dead_time=0, source_freq=1, override=False, ref=False): # Hz
    # this is the characterized ref matrix
    noisy_ref_dm = np.array([[0.94642857+0.j, 0.        +0.j], [0.        +0.j, 0.05357143+0.j]])

    if ref:
      f = 1
      p = 2 * f - 1
      noisy_state = .5 * np.array([
        [1 + p, 0],
        [0, 1 - p]])
    else:
      if override:
        noisy_state = noisy_ref_dm
      else:
        noisy_state = get_char_noisy_state(s_x, s_y, s_z)

    ref_state, = ns.qubits.create_qubits(1, no_state=True)

    ns.qubits.qubitapi.assign_qstate([ref_state], noisy_ref_dm)
    #ns.qubits.qubitapi.assign_qstate([ref_state], ks.s0)


    noisy_input1, = ns.qubits.create_qubits(1, no_state=True)

    # np.diag is a function to easily create diagonal matrices
    ns.qubits.qubitapi.assign_qstate([noisy_input1], noisy_state)

    print("input fidelity check:", ns.qubits.dmutil.dm_fidelity(ns.qubits.reduced_dm([noisy_input1]), ns.qubits.reduced_dm([ref_state]), dm_check=True, squared=True))
    #print("input fidelity check:", ns.qubits.fidelity([noisy_input1], ks.s0, squared=True))

    noisy_input_state = DenseDMRepr(noisy_state)

    emitter = Node("Emitter", qmemory=QuantumMemory("EmitterQmem", num_positions=1))
    qsource = QSource(f"emitter_qsource", StateSampler([noisy_input_state], [1]), num_ports=1,
                          status=SourceStatus.EXTERNAL, frequencey=source_freq)
    emitter.add_subcomponent(qsource, name="qsource")

    receiver = Node("Receiver")

    network = Network("raman_network")
    network.add_nodes([emitter, receiver])

    q_conn = QuantumConnection(length=node_distance, depolar_rate=depolar_rate)

    port_ac, port_bc = network.add_connection(emitter, receiver, connection=q_conn, label="quantum",
                           port_name_node1="qout_receiver", port_name_node2="qin_emitter")

    emitter.subcomponents["qsource"].ports['qout0'].forward_output(emitter.ports[port_ac])

    return network, noisy_input1


def get_dep_prob_direct(incident_photons_per_s):

  delta_t_gate = 1 # sec
  raman_photons_incident = incident_photons_per_s * delta_t_gate * 10**4

  quantum_photons_incident_per_s = 125 * 10**3 # 0.5 km

  depolar_prob = raman_photons_incident / (raman_photons_incident + quantum_photons_incident_per_s)
  print("dep prob", depolar_prob)
  return depolar_prob


def run_char_coex_ent_experiment(s_x, s_y, s_z, incident_photons_per_s=0, verbose=True, ref=False, override=False):

  coex_fiber_dm_data = pandas.DataFrame()

  seed = -1
  c = .0002

  seed += 1
  ns.set_random_state(seed=seed)
  ns.sim_reset()
  fibre_length = 1

  delay = (fibre_length / c) + 1

  depolar_prob = get_dep_prob_direct(incident_photons_per_s=incident_photons_per_s)

  network, noisy_input1 = characterized_network_setup(s_x, s_y, s_z, node_distance = fibre_length, depolar_rate = depolar_prob, dead_time=0, source_freq=1, ref=ref, override=override)


  node_e = network.get_node("Emitter")
  node_r = network.get_node("Receiver")

  emit_prot = EmitProtocol(node_e, delay = delay, iterations = 1)
  recv_prot = ReceiveProtocol(node_r, verbose=True )

  coex_fiber_dm = setup_datacollectors(emit_prot, recv_prot)

  emit_prot.start()
  recv_prot.start()

  ns.sim_run()

  # save data
  coex_fiber_dm = coex_fiber_dm.dataframe
  #print(df_fidelity.shape)
  # label this data with this run's seed
  coex_fiber_dm['iteration'] = seed
  # concatenate this run's data with the main fidelity data
  coex_fiber_dm_data = pandas.concat([coex_fiber_dm_data, coex_fiber_dm])

  coex_dark_fidelities = []
  coex_pure_fidelities = []

  # calculate fidelity

  b2 = coex_fiber_dm_data.iloc[0]['b2']

  noisy_ref_dm = np.array([[0.94642857+0.j, 0.        +0.j], [0.        +0.j, 0.05357143+0.j]])
  noisy_ref_state, = ns.qubits.create_qubits(1, no_state=True)

  ns.qubits.qubitapi.assign_qstate([noisy_ref_state], noisy_ref_dm)

  f = ns.qubits.dmutil.dm_fidelity(ns.qubits.reduced_dm([b2]), ns.qubits.reduced_dm([noisy_ref_state]), squared=True, dm_check=True)


  print("f", f)
  print("depolar_prob", depolar_prob)

  return f

def main():
    ns.set_qstate_formalism(ns.QFormalism.DM)

    verbose = True
    characterize_ref_state = False 
    if characterize_ref_state:
      ### characterize coexisting 1530 reference state 
      # the resulting density matrix is hard-coded in 
      # characterized_network_setup() as noisy_ref_dm
      fid_1530_ref = run_char_coex_ent_experiment(s_x=0, s_y=0, s_z=0, incident_photons_per_s=1.5, verbose=False, ref=False)
    else:
      ### calculate fidelity of the rest with respect to 1530
      #1
      coex_fid_1530 = run_char_coex_ent_experiment(s_x=.3, s_y=.3, s_z=.3, incident_photons_per_s=1.5, verbose=False, override=True)
      # 97
      coex_fid_1535 = run_char_coex_ent_experiment(s_x=.3, s_y=.16, s_z=.83, incident_photons_per_s=1.25, verbose=False)
      # 95
      coex_fid_1540 = run_char_coex_ent_experiment(s_x=0.37377046, s_y=.12, s_z=0.72, incident_photons_per_s=0.5, verbose=False)
      # 92
      coex_fid_1545 = run_char_coex_ent_experiment(s_x=0.37377046, s_y=.25, s_z=0.61, incident_photons_per_s=0.5, verbose=False)
      #90 s_x=0.35, s_y=.4, s_z=0.49
      coex_fid_1550 = run_char_coex_ent_experiment(s_x=0.16, s_y=.3, s_z=0.49, incident_photons_per_s=1.5, verbose=False)
      # 84
      coex_fid_1555 = run_char_coex_ent_experiment(s_x=0.3, s_y=.25, s_z=0.33, incident_photons_per_s=1.7, verbose=False)
      # 80
      coex_fid_1560 = run_char_coex_ent_experiment(s_x=0.38, s_y=.6, s_z=0.38, incident_photons_per_s=1.6, verbose=False)
      # 78
      coex_fid_1565 = run_char_coex_ent_experiment(s_x=0.5, s_y=.5, s_z=0.3, incident_photons_per_s=1.6, verbose=False)

      # Sample data
      powers = [1530, 1535, 1540, 1545, 1550, 1555, 1560, 1565]
      coex_fidelities = [coex_fid_1530, coex_fid_1535, coex_fid_1540, coex_fid_1545, coex_fid_1550, coex_fid_1555, coex_fid_1560, coex_fid_1565]
      expected = [1, .98, .94, .91, .90, .88, .81, .79]


      # Set Seaborn theme for better aesthetics
      sns.set_theme(style="whitegrid")

      # Create the plot
      plt.figure(figsize=(6, 4))  # Adjust figure size
      sns.lineplot(x=powers, y=coex_fidelities, marker="o", label="Simulated", linestyle="-", markersize=7)
      sns.lineplot(x=powers, y=expected, marker="^", label="Experimental", linestyle="--", markersize=7)


      # Add labels and title
      plt.title("Fidelity of Coexisting Direct Transmission")
      plt.xlabel("Quantum Channel Wavelength [nm]")
      plt.ylabel(r'$F(\rho_{dark}(1530), \rho_{coex}(\lambda))$')
      plt.ylim(0, 1)


      # Add legend
      plt.legend()

      # Show the plot
      plt.show()

if __name__ == "__main__":
  main()







