# All experimental values are taken from
# Thomas, J. M., Kanter, G. S., & Kumar, P. (2023). 
# Designing noise-robust quantum networks coexisting in the classical fiber infrastructure. 
# Optics Express, 31(26), 43035-43047.

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas

import netsquid as ns

from netsquid.protocols import NodeProtocol
from netsquid.protocols import Signals

from netsquid.nodes.connections import Connection
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components import QuantumChannel
from netsquid.components.models import FixedDelayModel
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
from netsquid.components.component import Message
from netsquid.components import QuantumMemory
from netsquid.qubits.dmtools import DenseDMRepr

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

        b1, = prot_emitter.node.qmemory.pop(0)
        b2 = prot_rx.bp

        return {"b1": b1,
                "b2": b2,
                "dm": ns.qubits.reduced_dm([b1, b2])}

    # init datacollector to call get_fidelity() when triggered
    dc_fidelity = DataCollector(get_fidelity, include_entity_name=False)
    # configure datacollector to trigger when Bob's Protocol signals SUCCESS
    dc_fidelity.collect_on(pydynaa.EventExpression(source=prot_rx,
                                          event_type=Signals.SUCCESS.value))

    return dc_fidelity


def get_noisy_emitted_state(emitted_fidelity, v_dark):
  p = 1 - emitted_fidelity

  # Define Bell states as column vectors
  phi_plus = (1/np.sqrt(2)) * np.array([[1, 0, 0, 1]]).T
  phi_minus = (1/np.sqrt(2)) * np.array([[1, 0, 0, -1]]).T
  psi_plus = (1/np.sqrt(2)) * np.array([[0, 1, 1, 0]]).T
  psi_minus = (1/np.sqrt(2)) * np.array([[0, 1, -1, 0]]).T

  # Compute the density matrices for each Bell state
  rho_phi_plus = phi_plus @ phi_plus.T
  rho_phi_minus = phi_minus @ phi_minus.T
  rho_psi_plus = psi_plus @ psi_plus.T
  rho_psi_minus = psi_minus @ psi_minus.T


  # Construct the final mixed state
  rho = (1 - p) * rho_phi_plus + (p / 3) * (rho_psi_plus + rho_psi_minus + rho_phi_minus)

  p_deph = .047 # this value yields a desired dark fidelity of .977 
  coherence_factor = np.exp(-p_deph)
  rho = np.array([
      [0.5, 0, 0, 0.5 * coherence_factor],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0.5 * coherence_factor, 0, 0, 0.5]
  ], dtype=complex)

  return rho



def characterized_network_setup(bell_state, emitted_fidelity, v_dark, node_distance=4e-3, depolar_rate=0, dead_time=0, source_freq=1):

    #werner_state = get_werner_state(emitted_fidelity, bell_state)
    noisy_emitted_state = get_noisy_emitted_state(emitted_fidelity, v_dark)


    pure1, pure2 = ns.qubits.create_qubits(2, no_state=True)
    if bell_state == 'phi+':
        ns.qubits.qubitapi.assign_qstate([pure1, pure2], ks.b00)
    elif bell_state == 'psi+':
        ns.qubits.qubitapi.assign_qstate([pure1, pure2], ks.b11)
    #print(ns.qubits.reduced_dm([pure1, pure2]))

    noisy_input1, noisy_input2 = ns.qubits.create_qubits(2, no_state=True)

    # np.diag is a function to easily create diagonal matrices
    ns.qubits.qubitapi.assign_qstate([noisy_input1, noisy_input2], noisy_emitted_state)

    #ns.qubits.qubitapi.assign_qstate([noisy_input1, noisy_input2], ks.b00)
    #ns.qubits.dephase(noisy_input1, prob=0.018)
    #ns.qubits.dephase(noisy_input2, prob=0.01)


    print("input fidelity check:", ns.qubits.dmutil.dm_fidelity(ns.qubits.reduced_dm([noisy_input1, noisy_input2]), ns.qubits.reduced_dm([pure1, pure2]), dm_check=True, squared=True))
    noisy_input_state = DenseDMRepr(noisy_emitted_state)

    emitter = Node("Emitter", qmemory=QuantumMemory("EmitterQmem", num_positions=1))
    qsource = QSource(f"emitter_qsource", StateSampler([noisy_input_state], [1]), num_ports=2,
                          status=SourceStatus.EXTERNAL, frequencey=source_freq)
    emitter.add_subcomponent(qsource, name="qsource")

    receiver = Node("Receiver")


    network = Network("raman_network")
    network.add_nodes([emitter, receiver])

    q_conn = QuantumConnection(length=node_distance, depolar_rate=depolar_rate)

    port_ac, port_bc = network.add_connection(emitter, receiver, connection=q_conn, label="quantum",
                           port_name_node1="qout_receiver", port_name_node2="qin_emitter")

    emitter.subcomponents["qsource"].ports['qout1'].connect(emitter.qmemory.ports['qin0'])
    emitter.subcomponents["qsource"].ports['qout0'].forward_output(emitter.ports[port_ac])

    print("Mixed state ρ:")
    print(ns.qubits.reduced_dm([noisy_input1, noisy_input2]))

    return network, noisy_input1, noisy_input2


def get_dep_prob_from_v(v_sig, v_idl):
  v_hv = v_sig
  v_dark = v_hv


  f_idl = (1 + 3 * v_idl) / 4
  f_sig = (1 + 3 * v_dark) / 4


  depolar_prob_sig = 1 - f_sig
  depolar_prob_idl = 1 - f_idl

  print("depolar_prob_sig", depolar_prob_sig)
  print("depolar_prob_idl", depolar_prob_idl)

  return depolar_prob_sig, depolar_prob_idl, f_sig, f_idl


def run_char_coex_ent_experiment(bell_state, emitted_fidelity, architecture, ram=0, raman=False, state=None, v_dark=0, v_sig=0, v_idl=0, verbose=True):

  coex_fiber_dm_data = pandas.DataFrame()

  seed = -1
  c = .0002

  seed += 1
  ns.set_random_state(seed=seed)
  ns.sim_reset()
  fibre_length = 1

  delay = (fibre_length / c) + 1

  depolar_prob_sig, depolar_prob_idl, f_sig, f_idl = get_dep_prob_from_v(v_sig=v_sig, v_idl=v_idl)

  network, noisy_input1, noisy_input2 = characterized_network_setup(bell_state, emitted_fidelity, node_distance = fibre_length, depolar_rate = depolar_prob_sig, dead_time=0, source_freq=1, v_dark=v_dark)

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

  # calculate fidelity
  b1 = coex_fiber_dm_data.iloc[0]['b1']
  b2 = coex_fiber_dm_data.iloc[0]['b2']

  f_ent = ns.qubits.dmutil.dm_fidelity(ns.qubits.reduced_dm([b1, b2]), ns.qubits.reduced_dm([noisy_input1, noisy_input2]), squared=True, dm_check=True)

  
  print("f_ent", f_ent)
  print("depolar_prob_sig", depolar_prob_sig)

  return f_ent




def main():
  ns.set_qstate_formalism(ns.QFormalism.DM)

  verbose = True

  # launch power
  f_ent = .977
  v_dark = 2 * f_ent - 1
  u_c = (1 / v_dark) - 1
  print("v_dark", v_dark)
  print("uc", u_c)

  print(v_dark)
  coex_fid_14 = run_char_coex_ent_experiment(bell_state="phi+", emitted_fidelity=f_ent, architecture="launch_power", v_sig=.985, v_dark = v_dark, ram=2000, verbose=True)

  print()
  coex_fid_16 = run_char_coex_ent_experiment(bell_state="phi+", emitted_fidelity=f_ent, architecture="launch_power", v_sig=.975, v_dark=v_dark, ram=3000, verbose=True)

  print()
  coex_fid_18 = run_char_coex_ent_experiment(bell_state="phi+", emitted_fidelity=f_ent, architecture="launch_power", v_sig = .96, v_dark=v_dark, ram=4000, verbose=True)
  

  powers = [14, 16.2, 18.1]
  coex_fidelities = [coex_fid_14, coex_fid_16, coex_fid_18]
  expected = [0.996, 0.991, 0.985]
  expected_errors = [0.0036, 0.0037, 0.0032]

  # Compute average difference
  diff_14 = abs(0.996 - coex_fid_14)
  diff_16 = abs(0.991 - coex_fid_16)
  diff_18 = abs(0.985 - coex_fid_18)
  avg_diff = (diff_14 + diff_16 + diff_18) / 3

  print("Average difference:", avg_diff)

  # Set Seaborn style
  sns.set_theme(style="whitegrid")

  # Create figure
  plt.figure(figsize=(6, 4))

  # Plot data with Seaborn
  sns.lineplot(x=powers, y=coex_fidelities, marker="o", label="Simulated", linestyle="-", markersize=7)
  sns.lineplot(x=powers, y=expected, marker="^", label="Experimental", linestyle="--", markersize=7)

  # Add error bars for expected values
  plt.errorbar(powers, expected, yerr=expected_errors, fmt='o', capsize=5, color='black')

  # Customize plot
  plt.title("Fidelity of Coexisting Entanglement Distribution")
  plt.xlabel("Classical Channel Launch Power [dBm]")
  plt.ylabel(r'$F(\rho_{dark}, \rho_{coex})$')
  plt.ylim(.95, 1)
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()
