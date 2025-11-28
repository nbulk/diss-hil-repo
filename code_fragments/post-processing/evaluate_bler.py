import numpy as np
import zc_sequence as zc
import matplotlib.pyplot as plt
import sionna.phy as sn
import create_payload as pg
import h5py as hdf
import scipy.signal as sci_signal
import pdb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", type = int, help="Set Tx-Gain", nargs='?',default=int(30))
parser.add_argument("-m", type = int, help ="Select MCS", nargs='?',default =int(5))

args = parser.parse_args()

mcs = args.m
num_different_frames = 1
tx_power = args.p

zc_length = 571

payload_generator = pg.MuSeC_payload_generator(mcs = mcs, batch_size=num_different_frames)
payload = payload_generator.generate_payload()

print(rx_fname)

# with hdf.File(rx_name, 'a') as file:

f = hdf.File(f'{rx_fname}', 'a')
#batch1 = np.array(f["batch1/rx_payload"])
#batch2 = np.array(f["batch2/rx_payload"])
#batch1 = np.array(f["rx_payload"])
batch1 = np.array(f["rx_frame"])
# f.close()

#rx_payload = np.vstack((batch1, batch2))
rx_payload = batch1

avg_energy_tx = np.mean(abs(payload)**2)
avg_energy_rx = np.mean(abs(rx_payload)**2, axis =1)
norm = avg_energy_tx/avg_energy_rx
norm = norm.reshape(-1,1)
rx_payload_norm = rx_payload* np.sqrt(norm)
rx_num_frames = norm.shape[0]


step = 2500
end = rx_num_frames//step
print(f'Number of Batches {end-1}')

bler_1 = []
bler_2 = []

for n in range(0, end-1):
    print(f"Starting with Batch: {n}")
    window = rx_payload_norm[n*step:(n+1)*step,:]
    llr_1, llr_2 = payload_generator.get_llr(rx_payload_norm[n*step:(n+1)*step,:])
    bits_1_hat = payload_generator.ldpc_decode(llr_1,payload_generator.service_1.decoder)
    bits_2_hat = payload_generator.ldpc_decode(llr_2,payload_generator.service_2.decoder)
    temp_bler_1 = payload_generator.calculate_bler(payload_generator.bits_1, bits_1_hat)
    temp_bler_2 = payload_generator.calculate_bler(payload_generator.bits_2, bits_2_hat)
    bler_1.append(temp_bler_1)
    bler_2.append(temp_bler_2)
    print(f'BLER 1 for Batch {n}: {temp_bler_1}')
    print(f'BLER 2 for Batch {n}: {temp_bler_2}')
   # plt.figure(300)
   # plt.scatter(window[1,:].real, window.imag[1,:])
   # plt.show()

bler_1_avg = np.mean(bler_1)
bler_2_avg = np.mean(bler_2)

print(bler_1_avg)
print(bler_2_avg)
    
dset = f.create_dataset('bler_1_batch', data= bler_1)
dset = f.create_dataset('bler_2_batch', data= bler_2)
dset = f.create_dataset('bler_1_avg', data= bler_1_avg)
dset = f.create_dataset('bler_2_avg', data= bler_2_avg)
f.close()
