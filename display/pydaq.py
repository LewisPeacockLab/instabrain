import pygame
import numpy as np
import ctypes
import yaml
import time
import os
import ni_consts as c
import game_filter as gf

#######################################
# generic driver and datatype imports #
#######################################

NIDAQ = ctypes.windll.nicaiu
int32 = ctypes.c_long
bool32 = ctypes.c_bool
uInt8 = ctypes.c_ubyte
uInt32 = ctypes.c_ulong
uInt64 = ctypes.c_ulonglong
float64 = ctypes.c_double
MINIMUM = float64(0)
MAXIMUM = float64(5)

class Pydaq(object):

    def __init__(self,
                 analog_in_chans='Dev5/ai0:1',
                 digital_out_chans='Dev5/port1/line0',
                 frame_rate=120,
                 lp_filt_freq=4,
                 lp_filt_order=3,
                 force_params='force_params.yml'):

        # specific variables for this device to operate with
        self.load_force_params(force_params)
        self.ai_chans = analog_in_chans
        self.do_chans = digital_out_chans
        self.task_handle = uInt32 
        self.ai_task_handle = self.task_handle(0)
        self.num_channels = 5
        self.points = 1
        self.points_read = uInt32()
        self.DAQ_data = np.zeros((self.points*self.num_channels), dtype=np.float64)

        NIDAQ.DAQmxCreateTask("", ctypes.byref(self.ai_task_handle))
        NIDAQ.DAQmxCreateAIVoltageChan(self.ai_task_handle, self.ai_chans, '',
                                           c.DAQmx_Val_Diff, MINIMUM, MAXIMUM,
                                           c.DAQmx_Val_Volts, None)
        NIDAQ.DAQmxStartTask(self.ai_task_handle)   

        self.do_task_handle = self.task_handle(0)
        self.points_write = uInt32()
        NIDAQ.DAQmxCreateTask('', ctypes.byref(self.do_task_handle))
        NIDAQ.DAQmxCreateDOChan(self.do_task_handle, self.do_chans, '',
            c.DAQmx_Val_ChanPerLine) # c.DAQmx_Val_ChanForAllLines
        NIDAQ.DAQmxStartTask(self.do_task_handle)   
        self.set_digital_out(0)

        # define data buffers and filters

        self.volts_zero = np.array([0.0,0.0,0.0,0.0,0.0])
        self.frame_rate = frame_rate
        self.zero_time = int(0.25*frame_rate)
        self.buffer_size = 3*frame_rate
        self.volts_buffer = np.zeros((5, self.buffer_size))
        self.force_buffer = np.zeros((5, self.buffer_size))
        (self.butter_lowpass_rt_b,
         self.butter_lowpass_rt_a) = gf.butter_lowpass(lp_filt_freq,
                                                       self.frame_rate,
                                                       lp_filt_order)


    def get_force(self):
        new_volts_in = self.filt_volts(self.get_volts())
        new_force_in = self.force_transform(new_volts_in-self.volts_zero)
        return new_force_in


    def get_volts(self):
        NIDAQ.DAQmxReadAnalogF64(self.ai_task_handle, uInt32(int(self.points)),
                                 float64(10.0),
                                 c.DAQmx_Val_GroupByChannel,
                                 self.DAQ_data.ctypes.data, 100,
                                 ctypes.byref(self.points_read), None)	
        return self.DAQ_data[0], self.DAQ_data[1], self.DAQ_data[2], self.DAQ_data[3], self.DAQ_data[4]


    def load_force_params(self, f='force_params.yml'):
        with open(f) as f:
            force_config = yaml.load(f)
        channel_data = force_config['channels']
        num_channels = np.size(channel_data)
        num_cal_points = np.size(channel_data[0]['volts'])
        self.force_params = np.zeros((2*num_channels,num_cal_points))
        for chan in range(np.size(channel_data)):
            self.force_params[2*chan,:] = channel_data[chan]['volts']
            self.force_params[2*chan+1,:] = channel_data[chan]['newtons']

    def force_transform(self, force_in):
        f_1 = self.force_interp(force_in[0], 0)
        f_2 = self.force_interp(force_in[1], 1)
        f_3 = self.force_interp(force_in[2], 2)
        f_4 = self.force_interp(force_in[3], 3)
        f_5 = self.force_interp(force_in[4], 4)
        self.force_buffer = np.roll(self.force_buffer, -1) 
        self.force_buffer[:,-1] = f_1, f_2, f_3, f_4, f_5
        return f_1, f_2, f_3, f_4, f_5


    def force_interp(self, force_in, axis):
        axis = 2*axis
        if force_in >= self.force_params[axis,-1]:
            force_out = self.force_params[axis+1,-1]
        elif force_in <= self.force_params[axis,0]:
            force_out = self.force_params[axis+1,0]
        else:
            idx = np.argmax(self.force_params[axis,:] > force_in) - 1
            force_out = (self.force_params[axis+1,idx]
                         + (self.force_params[axis+1,idx+1]-self.force_params[axis+1,idx])
                         /(self.force_params[axis,idx+1]-self.force_params[axis,idx])
                         *(force_in-self.force_params[axis,idx]))
        return force_out


    def filt_volts(self, volts_in):
        self.volts_buffer = np.roll(self.volts_buffer, -1) 
        self.volts_buffer[:,-1] = volts_in[0], volts_in[1], volts_in[2], volts_in[3], volts_in[4]
        v_1 = gf.filter_data_rt(self.volts_buffer[0,:],
                                self.butter_lowpass_rt_b,
                                self.butter_lowpass_rt_a)
        v_2 = gf.filter_data_rt(self.volts_buffer[1,:],
                                self.butter_lowpass_rt_b,
                                self.butter_lowpass_rt_a)
        v_3 = gf.filter_data_rt(self.volts_buffer[2,:],
                                self.butter_lowpass_rt_b,
                                self.butter_lowpass_rt_a)
        v_4 = gf.filter_data_rt(self.volts_buffer[3,:],
                                self.butter_lowpass_rt_b,
                                self.butter_lowpass_rt_a)
        v_5 = gf.filter_data_rt(self.volts_buffer[4,:],
                                self.butter_lowpass_rt_b,
                                self.butter_lowpass_rt_a)
        return v_1, v_2, v_3, v_4, v_5


    def set_volts_zero(self):
        self.volts_zero[0] = np.mean(self.volts_buffer[0,-self.zero_time:-1])
        self.volts_zero[1] = np.mean(self.volts_buffer[1,-self.zero_time:-1])
        self.volts_zero[2] = np.mean(self.volts_buffer[2,-self.zero_time:-1])
        self.volts_zero[3] = np.mean(self.volts_buffer[3,-self.zero_time:-1])
        self.volts_zero[4] = np.mean(self.volts_buffer[4,-self.zero_time:-1])

    def set_volts_zero_init(self):
        for time_step in range(self.zero_time+1):
            self.get_force()
        self.set_volts_zero()

    def set_digital_out(self, bool_out):
        data = np.array([bool_out], dtype=np.uint8)
        NIDAQ.DAQmxWriteDigitalLines(self.do_task_handle, int32(1),
            bool32(1), float64(10.0), c.DAQmx_Val_GroupByChannel,
            data.ctypes.data, ctypes.byref(int32(0)), None)
