#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
import pmt
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import threading
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-p", type = int, help="Set Tx-Gain")
parser.add_argument("-m", type = int, help ="Select MCS")
parser.add_argument("-f", type = int, help="Number of measured Frames")

args = parser.parse_args()


class transmit(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.tx_power = tx_power = args.p
        self.samp_rate = samp_rate = 1e6
        self.num_frame = num_frame = str(1)
        self.norm_tx_power = norm_tx_power = str(int(tx_power*100))
        self.mcs = mcs = str(args.m)
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0 = firdes.low_pass(1.0, samp_rate, 250e3, 10e3, window.WIN_BLACKMAN, 6.76)
        self.sps = sps = 24
        self.fname_rx = fname_rx = 'rx-'+mcs+'-'+num_frame+'-'+norm_tx_power+'-hil_data.dat'
        self.fname = fname = 'tx'+mcs+'-'+num_frame+'-hil_data.dat'

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("name=kermit", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('external', 0)
        self.uhd_usrp_source_0.set_time_source('external', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        _last_pps_time = self.uhd_usrp_source_0.get_time_last_pps().get_real_secs()
        # Poll get_time_last_pps() every 50 ms until a change is seen
        while(self.uhd_usrp_source_0.get_time_last_pps().get_real_secs() == _last_pps_time):
            time.sleep(0.05)
        # Set the time to PC time on next PPS
        self.uhd_usrp_source_0.set_time_next_pps(uhd.time_spec(int(time.time()) + 1.0))
        # Sleep 1 second to ensure next PPS has come
        time.sleep(1)

        self.uhd_usrp_source_0.set_center_freq(3.79e9, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_bandwidth(1e6, 0)
        self.uhd_usrp_source_0.set_normalized_gain(1, 0)
        self.uhd_usrp_sink_1 = uhd.usrp_sink(
            ",".join(("name=constantine", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,2)),
            ),
            "",
        )
        self.uhd_usrp_sink_1.set_clock_source('internal', 0)
        self.uhd_usrp_sink_1.set_samp_rate(samp_rate)
        _last_pps_time = self.uhd_usrp_sink_1.get_time_last_pps().get_real_secs()
        # Poll get_time_last_pps() every 50 ms until a change is seen
        while(self.uhd_usrp_sink_1.get_time_last_pps().get_real_secs() == _last_pps_time):
            time.sleep(0.05)
        # Set the time to PC time on next PPS
        self.uhd_usrp_sink_1.set_time_next_pps(uhd.time_spec(int(time.time()) + 1.0))
        # Sleep 1 second to ensure next PPS has come
        time.sleep(1)

        self.uhd_usrp_sink_1.set_center_freq(3.79e9, 0)
        self.uhd_usrp_sink_1.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_1.set_bandwidth(1e6, 0)
        self.uhd_usrp_sink_1.set_normalized_gain(0, 0)

        self.uhd_usrp_sink_1.set_center_freq(3.79e9, 1)
        self.uhd_usrp_sink_1.set_antenna("TX/RX", 1)
        self.uhd_usrp_sink_1.set_bandwidth(1e6, 1)
        self.uhd_usrp_sink_1.set_gain(tx_power, 1)
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_ccc(sps, variable_low_pass_filter_taps_0)
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.fir_filter_xxx_0 = filter.fir_filter_ccc(1, variable_low_pass_filter_taps_0)
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.blocks_null_source_0 = blocks.null_source(gr.sizeof_gr_complex*1)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, fname, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, fname_rx, False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.blocks_null_source_0, 0), (self.uhd_usrp_sink_1, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.uhd_usrp_sink_1, 1))
        self.connect((self.uhd_usrp_source_0, 0), (self.fir_filter_xxx_0, 0))


    def get_tx_power(self):
        return self.tx_power

    def set_tx_power(self, tx_power):
        self.tx_power = tx_power
        self.set_norm_tx_power(str(int(self.tx_power*100)))
        self.uhd_usrp_sink_1.set_gain(self.tx_power, 1)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_variable_low_pass_filter_taps_0(firdes.low_pass(1.0, self.samp_rate, 0.5e6, 10e3, window.WIN_BLACKMAN, 6.76))
        self.uhd_usrp_sink_1.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_num_frame(self):
        return self.num_frame

    def set_num_frame(self, num_frame):
        self.num_frame = num_frame
        self.set_fname('/home/niklas/Documents/git/hil_iq/2024-04-22-'+self.mcs+'-'+self.num_frame+'-seed-hil_data.dat')
        self.set_fname_rx('/home/niklas/Documents/git/hil_iq/2024-04-22-rx-'+self.mcs+'-'+self.num_frame+'-'+self.norm_tx_power+'-seed-hil_data.dat')

    def get_norm_tx_power(self):
        return self.norm_tx_power

    def set_norm_tx_power(self, norm_tx_power):
        self.norm_tx_power = norm_tx_power
        self.set_fname_rx('/home/niklas/Documents/git/hil_iq/2024-04-22-rx-'+self.mcs+'-'+self.num_frame+'-'+self.norm_tx_power+'-seed-hil_data.dat')

    def get_mcs(self):
        return self.mcs

    def set_mcs(self, mcs):
        self.mcs = mcs
        self.set_fname('/home/niklas/Documents/git/hil_iq/2024-04-22-'+self.mcs+'-'+self.num_frame+'-seed-hil_data.dat')
        self.set_fname_rx('/home/niklas/Documents/git/hil_iq/2024-04-22-rx-'+self.mcs+'-'+self.num_frame+'-'+self.norm_tx_power+'-seed-hil_data.dat')

    def get_variable_low_pass_filter_taps_0(self):
        return self.variable_low_pass_filter_taps_0

    def set_variable_low_pass_filter_taps_0(self, variable_low_pass_filter_taps_0):
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0
        self.fir_filter_xxx_0.set_taps(self.variable_low_pass_filter_taps_0)
        self.interp_fir_filter_xxx_0.set_taps(self.variable_low_pass_filter_taps_0)

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps

    def get_fname_rx(self):
        return self.fname_rx

    def set_fname_rx(self, fname_rx):
        self.fname_rx = fname_rx
        self.blocks_file_sink_0.open(self.fname_rx)

    def get_fname(self):
        return self.fname

    def set_fname(self, fname):
        self.fname = fname
        self.blocks_file_source_0.open(self.fname, True)




def main(top_block_cls=transmit, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()
    duration = args.f /1000
    time.sleep(duration*60)
    # try:
    #     input('Press Enter to quit: ')
    # except EOFError:
    #     pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
