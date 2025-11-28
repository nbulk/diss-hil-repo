import numpy as np
import sionna.phy as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

import service_generator as service


class MuSeC_payload_generator():
    
    def __init__(self, mcs, batch_size, seed=True):
        self.mcs = mcs
        self.batch_size = batch_size
        if seed == True:
            sn.config.seed = 1692

    def __read_mcs__(self):
        fname = f"0204_bler_data/mcs_config_{self.mcs}_bler_simulation.hdf"
        data_file = h5py.File(fname, 'r')
        self.modulation = np.array(data_file['Q_M']).astype(int)
        self.num_info_bits = np.array(data_file['k']).astype(int)
        self.num_codeword_bits = np.array(data_file['n']).astype(int)
        self.power_factor = np.array(data_file['power'])
    
    def generate_payload(self):
        self.__read_mcs__()
        self.service_1 = service.service_generator(int(self.modulation[0]), 
                                              self.batch_size, 
                                              self.num_info_bits[0], 
                                              self.num_codeword_bits[0])
        self.service_2 = service.service_generator(int(self.modulation[1]), 
                                              self.batch_size, 
                                              self.num_info_bits[1], 
                                              self.num_codeword_bits[1])
    
        msg_1 = self.service_1.generate_msg()
        msg_2 = self.service_2.generate_msg()
        self.messages = np.array([msg_1, msg_2])

        self.bits_1 = self.service_1.u
        self.bits_2 = self.service_2.u
    
        self.tx_payload = np.sqrt(self.power_factor[0])*msg_1 +np.sqrt(self.power_factor[1])*msg_2
        return np.array(self.tx_payload)
    
    def get_service_bits(self):
        return self.bits_1, self.bits_2
    
    def __demap_payload__(self, rx_payload):
        self.llr_1 = self.service_1.demapper(rx_payload,0.0)
        self.llr_1_hard = tf.where(self.llr_1 > 0, 1.0, 0.0)
        remapped_1 = self.service_1.mapper(self.llr_1_hard)
        self.remainder = (rx_payload - np.sqrt(self.power_factor[0])*remapped_1)/np.sqrt(self.power_factor[1])
        self.llr_2 = self.service_2.demapper(self.remainder,0.0)

    def get_llr(self, rx_payload):
        self.__demap_payload__(rx_payload)
        return self.llr_1, self.llr_2
    
    def ldpc_decode(self, llr, decoder):
        bits = decoder(llr)
        return bits

    def calculate_bler(self,bits, bits_hat):
        bler = sn.utils.compute_bler(bits, bits_hat)
        return bler

    def save_payload(self):
        fname = f"2025-04-18_{self.mcs}_transmitter.hdf"
        f = h5py.File(fname, 'w')
        dset = f.create_dataset("tx_symbols", data=np.array(self.tx_payload))
        dset = f.create_dataset("bits_1", data=np.array(self.bits_1))
        dset = f.create_dataset("bits_2", data=np.array(self.bits_2))
        f.close()

if __name__ == '__main__':
    payload_generator = MuSeC_payload_generator(mcs=16,batch_size=1)
    tx_payload = payload_generator.generate_payload()
    rx_payload = tx_payload
    llr_1 , llr_2 = payload_generator.get_llr(rx_payload)
    bits_1 = payload_generator.ldpc_decode(llr_1, payload_generator.service_1.decoder)
    bits_2 = payload_generator.ldpc_decode(llr_2, payload_generator.service_2.decoder)
    bler_1 = payload_generator.calculate_bler(payload_generator.bits_1, bits_1)
    bler_2 = payload_generator.calculate_bler(payload_generator.bits_2, bits_2)
