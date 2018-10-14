import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
import pandas as pd

fnames = ['1539470730_', '1539471192_', '1539471794_']

event_dfs = []
trigger_dfs = []
for f in fnames:
	event_dfs.append(pd.read_csv(f+'event.log'))
	trigger_dfs.append(pd.read_csv(f+'trigger.log'))
event_df = pd.concat(event_dfs)
trigger_df = pd.concat(trigger_dfs)

first_slice_times = np.array(event_df[event_df.event=='slc_18'].time)
last_slice_times = np.array(event_df[event_df.event=='mc_start'].time)
trigger_times = np.array(trigger_df.time)

sea.distplot(first_slice_times-trigger_times); plt.show()
