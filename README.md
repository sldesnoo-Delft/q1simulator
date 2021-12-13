# Q1Simulator
Simulator to execute Q1ASM and render the output channels.
The simulator can be used like a `pulsar_qcm` or `pulsar_qrm` object
of `qblox_instruments`.
It can be used with other Python software to test the
generation of Q1ASM program without using the actual hardware.
It can also be used stand-alone to check and display Q1ASM
waveforms-and-program files.

Q1ASM program and waveforms can be uploaded to the sequencer.
When the sequencer is started it executes the Q1ASM instructions
and renders the output to a numpy array with DAC output values.

The `acquire` instructions store the timestamp of the real-time
executor in the path data and increments the avg_cnt.
This makes it easy to check the timing of the acquire instructions.
The acquire instructions also check the time between the triggers
and report an error when there would be an FIFO error.

The complete Q1ASM instruction set has been implemented.
Q1Simulator simulates the (estimated) execution time of Q1ASM and
uses buffer between Q1Core and real-time executor. It aborts execution
when the real-time buffer would have an underrun.
The renderer uses the uploaded waveforms and the nco frequency.

# Example

    sim = Q1Simulator('q1sim')
    sim.sequencer0_waveforms_and_program('my_sequence.json')
    sim.arm_sequencer(0)
    sim.start_sequencer()
    sim.get_sequencer_state(0)
    sim.plot()
    sim.print_acquisitions()

# Q1ASM file viewer
`plot_q1asm_file` is a simple function that reads a file
and plots and prints the results.
    from q1simulator import plot_q1asm_file

    path = 'demo_data'
    plot_q1asm_file(path + r'/q1seq_P1.json')

`plot_q1asm_files` does the same for multiple files.
    from q1simulator import plot_q1asm_files, PlotDef

    path = 'demo_data'
    plot_q1asm_files([
            PlotDef(path + r'/q1seq_P1.json', 'P1', [0]),
            PlotDef(path + r'/q1seq_P2.json', 'P2', [1]),
            PlotDef(path + r'/q1seq_q1.json', 'q1', [2,3], lo_frequency=20e6),
            PlotDef(path + r'/q1seq_R1.json', 'R1', []),
            ])

See demo directory for some examples.

# Simulator rendering limits
The maximum length of the rendered output is limited to 2 ms,
because plots with more points do not perform well.
The maximum number of execution cycles is limited to 1e7,
because 1 cycle take 1 to 6 us in the simulator. So, after
10 to 60 seconds the simulator aborts execution of a sequencer.

The limits can be modified:
    sim.set('sequencer0_max_render_time', 10_000_000)
    sim.set('sequencer0_max_core_cycles', 1e9)


# Implemented methods
The following QCM/QRM methods and parameters are implemented by
the simulator and mimic behavior of the device:
- reset
- get_system_status
- get_sequencer_state
- arm_sequencer
- start_sequencer
- stop_sequencer
- get_acquisition_state
- get_acquisitions
- sequencer0_nco_freq
- sequencer0_mod_en_awg
- sequencer0_demod_en_acq
- sequencer0_channel_map_path0_out0_en
- sequencer0_channel_map_path1_out1_en
- sequencer0_channel_map_path0_out2_en
- sequencer0_channel_map_path1_out3_en
- sequencer0_waveforms_and_program

All other parameters only print the passed value.

# Simulator output
The simulator has some methods to show the simulator output.
- `plot()` shows pyplot charts with the rendered output.
- `print_acquisitions()` prints the path0 and path1 data and average counts.
- `print_registers()` prints the contents of the Q1 registers.

# Simulator logging
Q1Simulator has a logging feature to help with code debugging.
It uses of special comment line in the Q1ASM code.
The comment line should start with `#Q1Sim:` and is followed by the
simulator log command with the format log "message",register,options.
The options are:
* R: log register value
* T: log q1 and real-time executer time.
* F: format register value as Q1Pulse float

Example:

    #Q1Sim: log "t_wait",R3,TR
          wait  R3

Output:

    t_wait:          12 (0000000C) q1:   324 rt: 908 ns
    t_wait:          80 (00000050) q1:   588 rt: 1128 ns

# About Q1Simulator
One day after I had been working on the generation of Q1ASM I thought
it would be fun to write a interpreter to execute the generated Q1ASM.
A few hours later I had a parser, an interpreter and an output renderer
that plotted the output of the Q1ASM file that I had generated earlier
that day. Above all it showed there was an error in the timing of
generated signal. The next day I started to use Q1Simulator to
check the code I generated with Q1Pulse and pulselib. I found
several bugs in the generated code and in Q1Simulator.

Q1Simulator evolved into a useful tool.
Now it is available to all.

*Sander de Snoo*

