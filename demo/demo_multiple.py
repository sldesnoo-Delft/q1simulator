from q1simulator import plot_q1asm_files, PlotDef

path = 'demo_data'

plot_q1asm_files([
        PlotDef(path + r'/q1seq_P1.json', 'P1', [0]),
        PlotDef(path + r'/q1seq_P2.json', 'P2', [1]),
        PlotDef(path + r'/q1seq_q1.json', 'q1', [2,3], lo_frequency=100e6),
        PlotDef(path + r'/q1seq_R1.json', 'R1', []),
        ])
