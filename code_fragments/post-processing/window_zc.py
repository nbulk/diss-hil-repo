import numpy as np
import matplotlib.pyplot as plt
import zc_sequence as zc
import scipy.signal as sci_signal
import h5py as hdf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", type = int, help="Set Tx-Gain", nargs='?',const=int(5))
parser.add_argument("-m", type = int, help ="Select MCS", nargs='?',const =int(20))

args = parser.parse_args()

def windowed_zadoff_chu(windowed_rx, zc_seq, usf):
    peaks = np.empty(usf)
    zc_position = np.empty(usf)
    for start in range(1,usf):
        zc_correlated = abs(sci_signal.correlate(windowed_rx[start::usf],zc_seq, mode="full"))
        zc_peaks, _ = sci_signal.find_peaks(zc_correlated, height=0.9*max(zc_correlated))
        zc_peaks = zc_peaks[0]
        peaks[start-1] = zc_correlated[zc_peaks]
        zc_position[start-1] = zc_peaks
    return peaks, zc_position


def fine_sample_selection(windowed_rx_signal, zc_position, usf):
    start = zc_position - (3*usf)
    end = zc_position + (3*usf)
    # print(zc_position)

    region = windowed_rx_signal[start:end]
    try:
        peaks, _ = sci_signal.find_peaks(abs(region), prominence=0.7*max(abs(region)))
    except:
        peaks = [0]
        end_zc = 0
        print('Frame dropped')
    # plt.figure(3001)
    # plt.stem(abs(region))
    # # plt.plot(peaks, abs(region[peaks]),'rx')
    # plt.show()
    try:
        end_zc = start+peaks[-1]
    except:
        end_zc = 0
        print("Frame dropped")
    return end_zc

def delete_empty_frames(frame):
    print("Delete Frame")
    mask = np.where(np.average(frame, axis = 1) == 0)[0]
    frame_masked = np.delete(frame, mask, axis=0)
    delted_frames = len(mask)
    loss_ratio = delted_frames/len(frame)
    print(f"Number of undetected Frames: {delted_frames}")
    print(f"Frame loss Ration {loss_ratio*100}")
    return frame_masked, loss_ratio


usf = 24
frame_length = 2048
zero_padding_1_length = 10
primary_pilots_length = 40
secondary_pilots_length = 40
zc_length = 571
guard_interval_1 = 100
payload_len = 1024
guard_interval_2 = 103
guard_interval_3 = 160
padding_infront_zc_end = zero_padding_1_length+primary_pilots_length+zc_length

frame_len_dict = [zero_padding_1_length,primary_pilots_length,zc_length,guard_interval_1,payload_len,guard_interval_2,secondary_pilots_length,guard_interval_3]
for i in range(1, len(frame_len_dict)):
    frame_len_dict[i] = frame_len_dict[i-1] + frame_len_dict[i]

mcs = args.m
num_different_frames = 1
tx_power = args.p

zc_sequence = zc.generate_zc(length=zc_length, root=25)
tx_signal = np.fromfile(fname, dtype = np.complex64)

rx_fname
rx_signal = np.memmap(rx_fname, dtype=np.complex64)

num_frames = int(np.floor(rx_signal.shape[0]/(usf*frame_length)))

max_zc_list = []
end_zc_list = []
frame_start_list = []
frame_end_list = []
windowed_frame = np.zeros((num_frames, 2048), dtype=complex)

overlap_symbols = 450
window_size = int(usf*(frame_length+overlap_symbols))

start_idx_list = []
end_idx_list = []

start_idx = 0
for i in range(0, num_frames):
    end_idx = start_idx+window_size

    if end_idx < len(rx_signal)-usf*frame_length:
        start_idx_list.append(start_idx)
        end_idx_list.append(end_idx)
        start_idx = end_idx-(overlap_symbols*usf)



for n in range(10, num_frames-2):
    print(f"Process: {n/(num_frames-2)*100}")
    window_rx_signal = rx_signal[start_idx_list[n]:end_idx_list[n]]
    power = 10*np.log10(np.mean(np.abs(window_rx_signal)**2))
    print(f"Received Power: {power}")
    w_peaks, w_zc_position = windowed_zadoff_chu(window_rx_signal, zc_sequence,usf=usf)
    # print(w_peaks)
    # plt.show()
    max_zc = int(w_zc_position[np.argmax(w_peaks)])
    max_zc_list.append(max_zc)
    # max_zc = max_zc_list[0]
    if max_zc == 0:
        max_zc = max_zc_list[0]
    if max_zc > 1200:
        max_zc = max_zc_list[0]

    end_zc = fine_sample_selection(window_rx_signal, max_zc*usf ,usf)
    end_zc_list.append(end_zc)
    print(end_zc)


    frame_start = end_zc - (usf*padding_infront_zc_end)
    frame_end = frame_start + usf*frame_length
    frame_start_list.append(frame_start)
    frame_end_list.append(frame_end)
    print(frame_start)
    print(frame_end)

    if frame_start < 0:
        print("ping")
        windowed_frame[n,:] = 0
    else:
        try:
            windowed_frame[n,:] = window_rx_signal[frame_start:frame_end][::usf]
        except:
            print("Dropped Frame")
            windowed_frame[n,:] = 0


frame = np.copy(windowed_frame[:,1:])

frame_zeros = frame
frame, loss_ratio = delete_empty_frames(frame_zeros)

num_frames = frame.shape[0]
zero_padding_1 = frame[:,0:frame_len_dict[0]]
primary_pilots = frame[:,frame_len_dict[0]:frame_len_dict[1]]
rx_zc_seq = frame[:,frame_len_dict[1]:frame_len_dict[2]]
guard_1 = frame[:,frame_len_dict[2]:frame_len_dict[3]]
payload = frame[:,frame_len_dict[3]:frame_len_dict[4]]
guard_2 = frame[:,frame_len_dict[4]:frame_len_dict[5]]
secondary_pilots = frame[:,frame_len_dict[5]:frame_len_dict[6]]
guard_3 = frame[:,frame_len_dict[6]:frame_len_dict[7]]



pilot_value = np.array(1+0j)
secondary_phase_shift = np.zeros((num_frames,secondary_pilots_length), dtype=complex)
for frames in range(0, num_frames):
    for pilot in range(0, secondary_pilots_length-1):
        secondary_phase_shift[frames,pilot] = np.inner(pilot_value,secondary_pilots[frames,pilot])

avg_secondary_phase_shift = np.mean(secondary_phase_shift, axis =1)
avg_secondary_phase_shift = np.angle(avg_secondary_phase_shift,deg=False)
avg_secondary_phase_shift = np.mod(avg_secondary_phase_shift, 2*np.pi)
print(np.rad2deg(avg_secondary_phase_shift))
avg_secondary_phase_shift = avg_secondary_phase_shift.reshape(-1,1)

payload_phase_corrected = payload*np.exp(-1j*avg_secondary_phase_shift)

print("Post-Processing Done!")
