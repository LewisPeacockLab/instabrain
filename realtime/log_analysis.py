import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
import pandas as pd

backend_fnames = ['1539470730_', '1539471192_', '1539471794_']

backend_fnames = ['1539711187_']
backend_fnames = ['1539711801_']

event_dfs = []
trigger_dfs = []
for f in backend_fnames:
    event_dfs.append(pd.read_csv(f+'event.log'))
    trigger_dfs.append(pd.read_csv(f+'trigger.log'))
event_df = pd.concat(event_dfs)
trigger_df = pd.concat(trigger_dfs)

first_slice_times = np.array(event_df[event_df.event=='slc_18'].time)
last_slice_times = np.array(event_df[event_df.event=='mc_start'].time)
last_slice_times = last_slice_times
trigger_times = np.array(trigger_df.time)

sea.distplot(last_slice_times-trigger_times[:len(last_slice_times)]); plt.show()

display_frontend_fnames = ['1539544578_event.log', '1539545124_event.log', '1539545505_event.log']
display_backend_fnames = ['1539546265_event.log', '1539546675_event.log', '1539547052_event.log']

frontend_feedback_delays = np.array([])
for f in display_frontend_fnames:
    temp_df = pd.read_csv(f)
    trigger_df = temp_df[temp_df.event=='trigger']
    trigger_df = trigger_df[trigger_df.tr>4]
    feedback_df = temp_df[temp_df.event=='feedback']
    trigger_times = np.array(trigger_df.time)
    feedback_times = np.array(feedback_df.time)
    frontend_feedback_delays = np.append(frontend_feedback_delays, feedback_times-trigger_times)

backend_feedback_delays = np.array([])
for f in display_backend_fnames:
    temp_df = pd.read_csv(f)
    trigger_df = temp_df[temp_df.event=='trigger']
    trigger_df = trigger_df[trigger_df.tr>4]
    feedback_df = temp_df[temp_df.event=='feedback']
    trigger_times = np.array(trigger_df.time)
    feedback_times = np.array(feedback_df.time)
    backend_feedback_delays = np.append(backend_feedback_delays, feedback_times-trigger_times)

sea.distplot(frontend_feedback_delays); sea.distplot(backend_feedback_delays); plt.legend(['over network', 'localhost']); plt.show()
