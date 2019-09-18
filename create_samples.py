from pyo import *
import pyotools
import numpy as np
import pdb
import utils
import os, random


'''
Todo:  add an envolope, create model from pretrained something preferably
Might be just good to do random frequency so machine learning alg doesnt overfit to certain frequencies or get distracted by that because at thte end of the day thats a parameter that 
doesnt really matter that much.
'''


def create_train_dirs(sample_dir):
    '''
    Creates necessarry directories to store the train samples and their corresponding labels
    '''
    dirs_to_make = [os.path.join(sample_dir,"train/labels"), os.path.join(sample_dir, "train/samples"), os.path.join(sample_dir,"val/labels"),os.path.join(sample_dir,"val/samples"),\
        os.path.join(sample_dir,"test/labels"), os.path.join(sample_dir,"test/samples")]
    
    for d in dirs_to_make:
        if not os.path.exists(d):
            os.makedirs(d)


def create_oscsync_samples(param_dict,s, train_prob, val_prob, test_prob, sample_dir):

    '''
    Input: param_dict, a dictionary where each key is a parameter name and value is the list of parameters to iterate through. s: audio server
    train_prob: probability that a given sample should be in training set, val_prob and test_prob represent that same
    Output: list of of wav files and the corresponding synth parameters
    '''

    #first create output directories
    create_train_dirs(sample_dir)
    

    #First iterate through each type of table used
    table_list = param_dict['table']

    index_count = 0
    for table_index in range(len(table_list)):
        #each potential master,slave xfade, mul, add value
        for master_val in np.around(param_dict['master'],decimals=1):
            for slave_val in np.around(param_dict['slave'],decimals=1):
                for xfade_val in np.around(param_dict['xfade'], decimals=1):

                    # for mul_value in np.around(param_dict['mul'],decimals=1):
                    #     for add_Value in np.around(param['add'],decimals=1):
                    sosc=pyotools.OscSync(table_list[table_index],master_val.item(),slave_val.item(),xfade_val.item()).out()
                    # Path of the recorded sound file.

                    #now decide whether in train 
                    sample_dest = utils.train_val_test(train_prob, val_prob, test_prob)
                    label_path = ""
                    sample_path = ""
                    if(sample_dest == 0):
                        #goes to train set
                        label_path = os.path.join(sample_dir,"train/labels","synth" + str(index_count)+".npy")
                        sample_path = os.path.join(sample_dir,"train/samples","synth" + str(index_count)+".wav")
                        
                    elif(sample_dest == 1):
                        label_path = os.path.join(sample_dir,"val/labels","synth" + str(index_count)+".npy")
                        sample_path = os.path.join(sample_dir,"val/samples","synth" + str(index_count)+".wav")

                    else:
                        label_path = os.path.join(sample_dir,"test/labels","synth" + str(index_count)+".npy")
                        sample_path = os.path.join(sample_dir,"test/samples","synth" + str(index_count)+".wav")
                    
                    
                    # Record for 10 seconds a 24-bit wav file.
                    s.recordOptions(dur=10, filename=sample_path, fileformat=0, sampletype=1)
                   
                                        
                    s.recstart()
                    s.start()

                    #save the labels containing all the parameter info
                    labels_np = np.zeros(4)
                    
                    #table index order: saw, sqaure
                    labels_np[0] = table_index
                    labels_np[1] = master_val
                    labels_np[2] = slave_val
                    labels_np[3] = xfade_val
                    np.save(label_path,labels_np)
                    index_count+=1


def generate_samples(samples_path, label_path, batch_count):
    '''
    Generator function that loads wavs converts to spectrograms and finds labels to return as well
    Batch count is number of to return in each generator batch
    '''
    sample_count = 0
    X_train = None
    Y_train = None
    while(True):
        file = random.choice(os.listdir(samples_path))
        wav_path = os.path.join(samples_path, file)
        signal, sr = utils.load_audio(wav_path, mono=True)
        melgram = utils.make_melgram(signal, sr)
        file_name = file.split('.')[0]
        label = np.load(os.path.join(label_path,file_name+".npy"))
        if(X_train is None):
            #if first in sequence
            X_train = np.zeros((batch_count, melgram.shape[1], melgram.shape[2], melgram.shape[3]))
            Y_train = np.zeros((batch_count, label.shape[0]))
            X_train[0] = melgram[0]
            Y_train[0] = label
        else:
            X_train[sample_count] = melgram[0]
            Y_train[sample_count] = label
        
        if sample_count == batch_count-1:
            sample_count = 0
            yield X_train, Y_train
        else:
            sample_count += 1


                


            


                    
# def create_np_

                            
                            



# def create_attr_list(start,stop,div_size):
#     '''
#     Creates a list of 
#     '''
def freq_from_440(x,num_dec=2):
    '''
    Returns frequency of note x semitones away from 440
    '''
    return np.around(440*(2**(x/12)),decimals=num_dec)

def create_440_frequencies(num_create):
    '''
    Starts at 440 and exapnds num_create both directions to return list of common frequencies of musical notes
    '''
    freq_list = []

    #first add pitches below 440 Hz
    for i in range(num_create,0,-1):
        freq_list.append(freq_from_440(-i))
    
    #next add 440
    freq_list.append(440)

    #last add pitches above 440
    for i in range(1,num_create+1,1):
        freq_list.append(freq_from_440(i))
    
    return freq_list




# s = Server(audio="offline").boot()
# param_dict = {}
# param_dict['table'] = [SawTable().normalize(),SquareTable().normalize()]
# param_dict['master'] = create_440_frequencies(5)
# param_dict['slave'] = list(np.arange(100,1000,100))
# param_dict['xfade'] = list(np.around(np.arange(0,5,1),decimals=2))
# create_oscsync_samples(param_dict,s,.8,.1,.1, "/Users/jacob/sound_projects/ml-synth/data")
# generator = generate_samples("/Users/jacob/Desktop/train_wavs","/Users/jacob/Desktop/train_labels",10)

