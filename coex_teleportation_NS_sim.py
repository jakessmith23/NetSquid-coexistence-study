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
from netsquid.components.models import FixedDelayModel
import netsquid.qubits.ketstates as ks

from netsquid.components import QuantumChannel
from netsquid.components.models import DepolarNoiseModel
from netsquid.components.models import FibreDelayModel

from netsquid.qubits.dmtools import DenseDMRepr

from netsquid.nodes import Node
from netsquid.nodes import Network
from netsquid.components import QuantumMemory
from netsquid.qubits.dmtools import DenseDMRepr

from netsquid.util import DataCollector
from netsquid.protocols import Signals
import pydynaa

import pandas as pd
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


def get_noisy_emitted_state(emitted_fidelity):
  p = 1 - emitted_fidelity

  # Define Bell states as NumPy arrays
  psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
  psi_plus  = np.array([0, 1, 1, 0]) / np.sqrt(2)
  phi_plus  = np.array([1, 0, 0, 1]) / np.sqrt(2)
  phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)

  # Compute the density matrices for each Bell state
  rho_psi_minus = np.outer(psi_minus, psi_minus)
  rho_psi_plus  = np.outer(psi_plus, psi_plus)
  rho_phi_plus  = np.outer(phi_plus, phi_plus)
  rho_phi_minus = np.outer(phi_minus, phi_minus)


  # Construct the final mixed state
  rho = (1 - p) * rho_psi_minus + (p / 3) * (rho_psi_plus + rho_phi_plus + rho_phi_minus)

  p_deph = .047 # this value yields a desired dark fidelity of .977 
  coherence_factor = np.exp(-p_deph)
  rho = np.array([
      [0, 0, 0, 0],
      [0, 0.5, -0.5 * coherence_factor, 0],
      [0, -0.5 * coherence_factor, 0.5, 0],
      [0, 0, 0, 0]
  ], dtype=complex)


  # Print the result
  print("Mixed state ρ:")
  print(rho)
  return rho



def characterized_network_setup(bell_state, emitted_fidelity, node_distance=4e-3, depolar_rate=0, dead_time=0, source_freq=1): # Hz

    #werner_state = get_werner_state(emitted_fidelity, bell_state)
    noisy_emitted_state = get_noisy_emitted_state(emitted_fidelity)

    pure1, pure2 = ns.qubits.create_qubits(2, no_state=True)
    if bell_state == 'phi+':
        ns.qubits.qubitapi.assign_qstate([pure1, pure2], ks.b00)
    elif bell_state == 'psi-':
        ns.qubits.qubitapi.assign_qstate([pure1, pure2], ks.b11)

    #print(ns.qubits.reduced_dm([pure1, pure2]))

    noisy_input1, noisy_input2 = ns.qubits.create_qubits(2, no_state=True)

    # np.diag is a function to easily create diagonal matrices
    ns.qubits.qubitapi.assign_qstate([noisy_input1, noisy_input2], noisy_emitted_state)

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

    emitter.subcomponents["qsource"].ports['qout0'].connect(emitter.qmemory.ports['qin0'])
    emitter.subcomponents["qsource"].ports['qout1'].forward_output(emitter.ports[port_ac])

    #receiver.ports[port_bc].forward_input(receiver.qmemory.ports['qin0'])


    return network, noisy_input1, noisy_input2


def get_entr_distr_dep_prob(raman):
  detection_window = 0.5 # ns

  dark_counts_per_gate = 100 * 1e-9 * detection_window

  detection_efficiency = .90

  n_d0_db = -7.41 # dB
  n_d0 = 10**(n_d0_db / 10)
  print("n_d0", n_d0)
  mean_noise_photons_per_interval_d0 = 0
  mean_noise_photons_per_interval_d3 = 0


  n_d3_db = -12.88
  n_d3 = 10**(n_d3_db / 10)
  print("n_d3", n_d3)


  n_ac_db = -10.81 # dB
  n_ac = 10**(n_ac_db / 10)
  n_bc_db = -10.64 # dB
  n_bc = 10**(n_bc_db / 10)

  print("n_ac", n_ac)
  print("n_bc", n_bc)


  mean_photons_per_pulse_a = .018
  mean_photons_per_pulse_b = 0.013

  mean_noise_photons_per_pulse_a = .0027
  mean_noise_photons_per_pulse_b = .0022


  # total noise count probability per heradling arm detector
  nr_a_db = -5.1
  nr_a = 10**(nr_a_db / 10)
  nr_b_db = -5
  nr_b = 10**(nr_b_db / 10)

  herald_eff_before_trans_n0 = .19
  r_0 = (mean_noise_photons_per_interval_d0 * n_d0 * nr_a) + (mean_noise_photons_per_pulse_a * herald_eff_before_trans_n0) + dark_counts_per_gate

  print("big_R_0", r_0)

  herald_eff_before_trans_n3 = 0.05
  r_3 = (mean_noise_photons_per_interval_d3 * n_d3 * nr_b) + (mean_noise_photons_per_pulse_b * herald_eff_before_trans_n3) + dark_counts_per_gate

  print("big_R_3", r_3)

  n_r_a = None
  if raman:
    photons_per_ns = 10**5 * 1e-9 # cps to ns

    n_r_a = photons_per_ns * detection_window
  else:
    n_r_a = 0
  print("n_r_a", n_r_a)

  n_d1_db = -5.65 # dB
  n_d1 = 10**(n_d1_db / 10)
  print("n_d1", n_d1)

  n_r_b = None
  if raman:
    photons_per_ns = 10**5 * 1e-9 # cps/mW * mW

    n_r_b = photons_per_ns * detection_window  # arriving noise photons
  else:
    n_r_b = 0
  n_d2_db = -6.71 # dB
  n_d2 = 10**(n_d2_db / 10)
  print("n_r_b", n_r_b)
  print("n_d2", n_d2)

  charlie_insertion_loss_db = -1.2 # dB
  charlie_insertion_loss = 10**(charlie_insertion_loss_db / 10)

  n_R_a = (n_r_a * charlie_insertion_loss) + (mean_noise_photons_per_pulse_a * n_ac)
  n_R_b = (n_r_b * charlie_insertion_loss) + (mean_noise_photons_per_pulse_b * n_bc)

  print("n_R_a", n_R_a)
  print("n_R_b", n_R_b)



  # interference --> teleportation
  r_1 = ((n_d1 * (n_R_a + n_R_b)) / 2) + dark_counts_per_gate
  r_2 = ((n_d2 * (n_R_a + n_R_b)) / 2) + dark_counts_per_gate


  print("r_1, r_2", r_1, r_2)

  s_0 = (herald_eff_before_trans_n0 * mean_photons_per_pulse_a) + r_0
  s_3 = (herald_eff_before_trans_n3 * mean_photons_per_pulse_b) + r_3

  herald_eff_before_trans_n1 = .07
  herald_eff_before_trans_n2 = 0.09
  s_1 = (herald_eff_before_trans_n1 * mean_photons_per_pulse_a) + r_1
  s_2 = (herald_eff_before_trans_n2 * mean_photons_per_pulse_b) + r_2

  print("s:", s_0, s_1, s_2, s_3)

  c_01 = herald_eff_before_trans_n0 * herald_eff_before_trans_n1 * (mean_photons_per_pulse_a + mean_photons_per_pulse_a**2) + (s_0 * s_1)
  a_01 = s_0 * s_1

  c_23 = herald_eff_before_trans_n2 * herald_eff_before_trans_n3 * (mean_photons_per_pulse_b + mean_photons_per_pulse_b**2) + (s_2 * s_3)
  a_23 = (s_2 * s_3)


  print("c_01", c_01)
  print("a_01", a_01)

  v_ent_01 = (c_01 - a_01) / (c_01 + a_01)
  print("v_ent 01", v_ent_01)

  v_ent_23 = (c_23 - a_23) / (c_23 + a_23)
  print("v_ent 23", v_ent_23)

  #v_ent_avg = (v_ent_01 + v_ent_23) / 2
  #print("v_ent_avg", v_ent_avg)

  f_ent = (1 + 3 * v_ent_23) / 4

  #f_alice_physical = (1 + 3 * v_ent_01) / 4
  f_alice_physical = (1 + v_ent_01) / 2

  print("f_ent", f_ent)
  print("f_alice_physical", f_alice_physical)

  #depolar_prob = (4/3) * (1 - f_ent)
  depolar_prob_bob = 1 - f_ent
  depolar_prob_alice = 1 - f_alice_physical

  return depolar_prob_alice, depolar_prob_bob, f_ent, f_alice_physical


def run_char_coex_tele_experiment(bell_state, emitted_fidelity, raman, state, verbose=True):

  coex_fiber_dm_data = pandas.DataFrame()

  seed = -1
  c = .0002

  seed += 1
  ns.set_random_state(seed=seed)
  ns.sim_reset()
  fibre_length = 1

  delay = (fibre_length / c) + 1

  depolar_prob_alice, depolar_prob_bob, f_ent, f_alice_physical = get_entr_distr_dep_prob(raman)

  network, noisy_input1, noisy_input2 = characterized_network_setup(bell_state, emitted_fidelity, node_distance = fibre_length, depolar_rate = depolar_prob_bob, dead_time=0, source_freq=1)

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

  b1 = coex_fiber_dm_data.iloc[0]['b1']
  b2 = coex_fiber_dm_data.iloc[0]['b2']


  qubit_x, = ns.qubits.create_qubits(1, no_state=True)
  if state == "0":
    ns.qubits.qubitapi.assign_qstate([qubit_x], ks.s0)
  elif state =="1":
    ns.qubits.qubitapi.assign_qstate([qubit_x], ks.s1)
  elif state =="+":
    ns.qubits.qubitapi.assign_qstate([qubit_x], ks.h0)
  elif state =="-":
    ns.qubits.qubitapi.assign_qstate([qubit_x], ks.h1)

  # alice q depolarized
  f_a_init = 0.9898107208409896

  #p = 2 * f_a_init - 1
  #noisy_state = .5 * np.array([
  #  [1 + p, 0],
  #  [0, 1 - p]])
  p = 2 * (1 - f_a_init)
  noisy_state = .5 * np.array([
    [1, 1 - p],
    [1 - p, 1]])
  ns.qubits.qubitapi.assign_qstate([qubit_x], noisy_state)
  #ns.qubits.qubitapi.assign_qstate([qubit_x], ks.h0)


  ref_state, = ns.qubits.create_qubits(1, no_state=True)
  ns.qubits.qubitapi.assign_qstate([ref_state], ks.h0)

  print("alice fidelity check:", ns.qubits.dmutil.dm_fidelity(ns.qubits.reduced_dm([qubit_x]), ns.qubits.reduced_dm([ref_state]), dm_check=True, squared=True))

  print(ns.get_qstate_formalism())
  ns.qubits.depolarize(qubit_x, prob=depolar_prob_alice)

  if state == "0":
    f_alice = ns.qubits.fidelity([qubit_x], ks.s0, squared=True)
  elif state == "1":
    f_alice = ns.qubits.fidelity([qubit_x], ks.s1, squared=True)
  elif state == "+":
    f_alice = ns.qubits.fidelity([qubit_x], ks.h0, squared=True)
  elif state == "-":
    f_alice = ns.qubits.fidelity([qubit_x], ks.h1, squared=True)

  meas, prob = ns.qubits.gmeasure([qubit_x, b1], meas_operators=ns.qubits.operators.BELL_PROJECTORS)
  labels_bell = ("|00>", "|01>", "|10>", "|11>")

  gm_alice_X = int(labels_bell[meas][1])
  gm_alice_B = int(labels_bell[meas][2])

  if labels_bell[meas] == "|00>":
    print("here1")
    ns.qubits.operate(b2, ns.Z)
    ns.qubits.operate(b2, ns.X)
  if labels_bell[meas] == "|10>":
    print("here2")
    ns.qubits.operate(b2, ns.X)
  elif labels_bell[meas] == "|01>":
    print("here3")
    ns.qubits.operate(b2, ns.Z)
  elif labels_bell[meas] == "|11>":
    print("here4")
    pass

  print(ns.qubits.reduced_dm([b2]))


  if state == "0":
    f_teleported = ns.qubits.fidelity([b2], ks.s0, squared=True)
  elif state == "1":
    f_teleported = ns.qubits.fidelity([b2], ks.s1, squared=True)
  elif state == "+":
    f_teleported = ns.qubits.fidelity([b2], ks.h0, squared=True)
  elif state == "-":
    f_teleported = ns.qubits.fidelity([b2], ks.h1, squared=True)


  f_poles = .5 + (4/3) * (f_ent - .25) * (f_alice - 0.5)

  #v_hom = .842   # SiMULATE dark
  v_hom = .803 # coex
  f_eq = .5 + (4/3) * v_hom * (f_ent - .25) * (f_alice - 0.5)
  #f_avg = (1/3) * f_poles + (2/3) * f_eq
  f_avg = .5 + (8/9) * (v_hom + .5) * (f_ent - .25) * (f_alice - .5)


  print("f_alice qubit depolarized", f_alice)
  print("f_ent", f_ent)
  print("depolar_prob_alice", depolar_prob_alice)
  print("depolar_prob_bob", depolar_prob_bob)
  print("f_teleported", f_teleported)
  print("f_poles", f_poles)
  print("f_eq", f_eq)
  print("f_avg", f_avg)


  return f_poles

def main():
    ns.set_qstate_formalism(ns.QFormalism.DM)

    verbose = True

    #fid = run_char_coex_tele_experiment(bell_state="psi-", emitted_fidelity=1, #raman=False, state="0", verbose=True)

    #fid = run_char_coex_tele_experiment(bell_state="psi-", emitted_fidelity=0.#9773464333723751, raman=True, state="0", verbose=True)

    #fid = run_char_coex_tele_experiment(bell_state="psi-", emitted_fidelity=1, #raman=False, state="+", verbose=True)

    fid = run_char_coex_tele_experiment(bell_state="psi-", emitted_fidelity=0.9773464333723751, raman=True, state="+", verbose=True)


    #fid = run_char_coex_tele_experiment(bell_state="psi-", emitted_fidelity=1, raman=False, state="-", verbose=True)


    #fid = run_char_coex_tele_experiment(bell_state="psi-", emitted_fidelity=0.9773464333723751, raman=True, state="-", verbose=True)

    
   
if __name__ == "__main__":
  main()