import ctypes

##############################
# Setup some typedefs and constants
# to correspond with values in
# C:\Program Files\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include\NIDAQmx.h
# the typedefs
# the constants
DAQmx_Val_Cfg_Default = ctypes.c_long(-1)
DAQmx_Val_Volts = 10348
DAQmx_Val_Rising = 10280
DAQmx_Val_GroupByChannel = 0
DAQmx_Val_Volts = 10348
DAQmx_Val_GroupByScanNumber = 1
DAQmx_Val_Hz = 10373 # Hz
DAQmx_Val_High = 10192 # High
DAQmx_Val_Low = 10214 # Low
# Value for the Timeout parameter of DAQmxWaitUntilTaskDone
DAQmx_Val_WaitInfinitely = ctypes.c_double(-1)
# Values for DAQmx_Write_RegenMode ***
# Value set RegenerationMode1 ***
DAQmx_Val_AllowRegen = 10097        # Allow Regeneration
DAQmx_Val_DoNotAllowRegen = 10158   # Do Not Allow Regeneration
# Values for the Line Grouping parameter of DAQmxCreateDIChan and DAQmxCreateDOChan ***
DAQmx_Val_ChanPerLine = 0   # One Channel For Each Line
DAQmx_Val_ChanForAllLines = ctypes.c_uint32(1)   # One Channel For All Lines
# Values for DAQmx_SampQuant_SampMode ***
# Value set AcquisitionType ***
DAQmx_Val_FiniteSamps = 10178 # Finite Samples
DAQmx_Val_ContSamps = 10123 # Continuous Samples
DAQmx_Val_HWTimedSinglePoint = 12522 # Hardware Timed Single Point
# Values for DAQmx_AI_TermCfg ***
# Value set InputTermCfg ***
DAQmx_Val_RSE = 10083 # RSE
DAQmx_Val_NRSE = 10078 # NRSE
DAQmx_Val_Diff = 10106 # Differential
DAQmx_Val_PseudoDiff = 12529 # Pseudodifferential
# Values for the Signal Modifiers parameter of DAQmxConnectTerms
DAQmx_Val_DoNotInvertPolarity = 0   # Do not invert polarity
DAQmx_Val_InvertPolarity = 1   # Invert polarity
# Values for DAQmx_CI_Freq_Units
DAQmx_Val_Hz = 10373 # Hz
DAQmx_Val_Ticks = 10304 # Ticks
DAQmx_Val_FromCustomScale = 10065 # From Custom Scale

# Values for DAQmx_StartTrig_DelayUnits ***
# Value set DigitalWidthUnits1 ***
DAQmx_Val_SampClkPeriods = 10286 # Sample Clock Periods
DAQmx_Val_Seconds = 10364 # Seconds

DAQmx_Val_Rising = 10280 # Rising
DAQmx_Val_Falling = 10171 # Falling
# Values for DAQmx_CI_Freq_MeasMeth
DAQmx_Val_LowFreq1Ctr = 10105 # Low Frequency with 1 Counter
DAQmx_Val_HighFreq2Ctr = 10157 # High Frequency with 2 Counters
DAQmx_Val_LargeRng2Ctr = 10205 # Large Range with 2 Counters

#*** Value for the Number of Samples per Channel parameter of DAQmxReadAnalogF64, DAQmxReadBinaryI16, DAQmxReadBinaryU16,
#    DAQmxReadBinaryI32, DAQmxReadBinaryU32, DAQmxReadDigitalU8, DAQmxReadDigitalU32,
#    DAQmxReadDigitalLines, DAQmxReadCounterF64, DAQmxReadCounterU32 and DAQmxReadRaw ***
DAQmx_Val_Auto = -1
