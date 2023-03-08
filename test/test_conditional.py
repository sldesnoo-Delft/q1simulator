# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 19:04:38 2022

@author: sdesnoo
"""
from q1simulator import Cluster
import qcodes as qc

def init():
    qc.Instrument.close_all()
    cluster = Cluster('test', {2:'QRM'})
    qrm = cluster.module2
    qrm.config('trace', True)
    return cluster

def run(cluster, program, waveforms={}, weights={}, acquisitions={}):

    qrm = cluster.module2
    seq = qrm.sequencers[0]
    seq.channel_map_path0_out0_en = True
    seq.thresholded_acq_trigger_en(True)
    seq.thresholded_acq_trigger_address(1)
    seq.trigger1_count_threshold(1)
    seq.upload({
            'program': program,
            'waveforms': waveforms,
            'weights': weights,
            'acquisitions': acquisitions})
    qrm.arm_sequencer(0)

    for seq_number,trigger_number in [(1, 2), (2,3)]:
        seq = qrm.sequencers[seq_number]
        seq.thresholded_acq_trigger_en(True)
        seq.thresholded_acq_trigger_address(trigger_number)
        seq.upload({
                'program': 'stop',
                'waveforms': {},
                'weights': {},
                'acquisitions': {}
                })
        qrm.arm_sequencer(seq_number)

    cluster.start_sequencer()
    qrm.plot()

waveforms = {
    "gauss80": {
        "data": [
            0.00021073734644620426, 0.00032175475915178565, 0.000485954999850973, 0.0007260302794298448,
            0.0010730031959653244, 0.0015686819551156438, 0.0022685919639534573, 0.003245379444956216,
            0.004592637591638869, 0.0064290451339909255, 0.008902631069565252, 0.012194889831651198,
            0.01652437299454718, 0.022149284470989822, 0.02936851692650569, 0.038520501607367155,
            0.04997921782427386, 0.06414673848332593, 0.08144178947697502, 0.10228398474484361,
            0.1270736700907338, 0.156167662882968, 0.1898515957906482, 0.22831003268047262,
            0.27159598392359746, 0.31960185739844454, 0.3720341859374524, 0.42839461801732714,
            0.48796960000609346, 0.5498308842335518, 0.6128484583099875, 0.6757167248755714,
            0.7369938133110109, 0.7951528488218665, 0.8486429340866658, 0.8959566209925681,
            0.9356998720224928, 0.9666600271750343, 0.9878671723140003, 0.9986445826034864,
            0.9986445826034864, 0.9878671723140003, 0.9666600271750343, 0.9356998720224928,
            0.8959566209925681, 0.8486429340866658, 0.7951528488218665, 0.7369938133110109,
            0.6757167248755714, 0.6128484583099875, 0.5498308842335518, 0.48796960000609346,
            0.42839461801732714, 0.3720341859374524, 0.31960185739844454, 0.27159598392359746,
            0.22831003268047262, 0.1898515957906482, 0.156167662882968, 0.1270736700907338,
            0.10228398474484361, 0.08144178947697502, 0.06414673848332593, 0.04997921782427386,
            0.038520501607367155, 0.02936851692650569, 0.022149284470989822, 0.01652437299454718,
            0.012194889831651198, 0.008902631069565252, 0.0064290451339909255, 0.004592637591638869,
            0.003245379444956216, 0.0022685919639534573, 0.0015686819551156438, 0.0010730031959653244,
            0.0007260302794298448, 0.000485954999850973, 0.00032175475915178565, 0.00021073734644620426],
        "index": 0
        },
    "zero80": {
        "data": [0.0]*80,
        "index": 1
        },
    }

sim = init()

#%%
sim.reset()
program='''
wait_sync 100
latch_en 1,4
#Q1Sim:sim_trigger 1, 1

play 0,0,92 # subtract 8 for next conditional

set_cond 1, 1, 0, 4
wait 8 # 4ns * number of statements including this wait
play 0,1,72 # subtract 8 ns for next conditional
# duration of else: 8 ns

set_cond 1, 2, 0, 4
play 1,0,60
wait 20

set_cond 0, 0, 0, 0
play 1,1,100
stop
'''

run(sim, program, waveforms)

print('====')

program='''
wait_sync 100
latch_en 1,4
#Q1Sim:sim_trigger 1, 1
play 0,0,84 # subtract 16ns: total of else_wait of if/elif conditionals (final else not included)

set_cond 1, 1, 0, 4
wait 16 # total else_wait of if/elif conditionals
play 0,1,68 # subtract 12 ns: total of else of next conditionals
# duration of else: 8 ns

set_cond 1, 2, 0, 4
wait 8 # wait 8 ns: # total else_wait of if/elif conditionals - total else_wait previous conditionals
play 1,0,76 # subtract 4 ns for else of next conditional
# duration of else: 8 ns

set_awg_gain 32000,16000
set_cond 1, 4, 0, 4
play 1,1,80 # starts with 16 ns delay
# duration of else: 4 ns

set_cond 0, 0, 0, 0
play 0,0,100
stop
'''

run(sim, program, waveforms)
