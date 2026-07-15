from presto import lockin

#### File parameters
save_folder = "/home/nanophys-meas/Desktop/Jai Master Thesis"
sample = "LKIPA_4_08-23"
save_file = "LKIPA_4"

# save_folder = "C:/Users/Admin/Desktop/QAFM/"
# sample = "LKIPA_2_06-09"
# save_file = "LKIPA_2.hdf5"

### Vivace/Presto parameters

# BOX = "bravo"
# BOX_ADDRESS = "192.168.88.51"

# """Vivace converter rates"""
# CONVERTER_RATES = dict(
#     adc_mode=lockin.AdcMode.Mixed,
#     adc_fsample=lockin.AdcFSample.G3_2,
#     dac_mode=lockin.DacMode.Mixed02,
#     dac_fsample=lockin.DacFSample.G6_4,
#     )
# v_input_port = 1 # Physical input port for Presto (delta)
# v_output_signal_port = 1  # Physical output port for presto (delta)
# vdc_port = 2

# """Presto converter rates"""
# BOX = "delta"
# BOX_ADDRESS = "192.168.88.53"
# PORT = None

## For running remote
BOX = "delta"
BOX_ADDRESS = "130.237.35.90"
PORT = 42873
    
CONVERTER_RATES = dict(
    adc_mode=lockin.AdcMode.Mixed,
    adc_fsample=lockin.AdcFSample.G2,
    dac_mode=[lockin.DacMode.Mixed04, lockin.DacMode.Mixed02, lockin.DacMode.Mixed02, lockin.DacMode.Mixed02],
    dac_fsample=[lockin.DacFSample.G10, lockin.DacFSample.G6, lockin.DacFSample.G6, lockin.DacFSample.G6]
)
# CONVERTER_RATES = dict(
#     adc_mode=lockin.AdcMode.Mixed,
#     adc_fsample=lockin.AdcFSample.G2,
#     dac_mode=lockin.DacMode.Mixed02,
#     dac_fsample=lockin.DacFSample.G6
# )
v_input_port = 5 # Physical input port for Vivace
v_output_signal_port = 8  # Physical signal output port for Vivace
v_output_pump_port = 2 # Physical pump output port for Vivace
vdc_port=2

#### Measurement params
Instr = "Presto"
cryostat = "DR"

temperature = 10e-3
atten = 60 + 20
rt_gain_output = 15
rt_gain_input = 20

# Instr = "Vivace"
# cryostat = "PPMS"

# temperature = 0
# atten = 60
# rt_gain_output = 0
# rt_gain_input = 0

comment = "Pump_port_connected_DC_port_connected_with_150kOhm&Cold_LP_filter"
# comment = "Pump_port_connected_DC_port_disconnected"