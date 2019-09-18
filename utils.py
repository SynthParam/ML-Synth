import librosa
import numpy as np
import random
import glob



def load_audio(audio_path, mono=None, sr=None, convertOSXaliases=True):  # wrapper for librosa.load
    try:
        signal, sr = librosa.load(audio_path, mono=mono, sr=sr)
    except NoBackendError as e:
        if ('Darwin' == platform.system()):   # handle OS X alias files gracefully
            source = resolve_osx_alias(audio_path, convert=convertOSXaliases, already_checked_os=True) # convert to symlinks for next time
            try:
                signal, sr = librosa.load(source, mono=mono, sr=sr)
            except NoBackendError as e:
                print("\n*** ERROR: Could not open audio file {}".format(audio_path),"\n",flush=True)
                raise e
        else:
            print("\n*** ERROR: Could not open audio file {}".format(audio_path),"\n",flush=True)
            raise e
    return signal, sr


def make_melgram(mono_sig, sr, n_mels=128):   # @keunwoochoi upgraded form 96 to 128 mel bins in kapre
    #melgram = librosa.logamplitude(librosa.feature.melspectrogram(mono_sig,  # latest librosa deprecated logamplitude in favor of amplitude_to_db
    #    sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

    melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(mono_sig,
        sr=sr, n_mels=n_mels))[np.newaxis,:,:,np.newaxis]     # last newaxis is b/c tensorflow wants 'channels_last' order

    '''
    # librosa docs also include a perceptual CQT example:
    CQT = librosa.cqt(mono_sig, sr=sr, fmin=librosa.note_to_hz('A1'))
    freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=librosa.note_to_hz('A1'))
    perceptual_CQT = librosa.perceptual_weighting(CQT**2, freqs, ref=np.max)
    melgram = perceptual_CQT[np.newaxis,np.newaxis,:,:]
    '''
    return melgram

def train_val_test(train_prob, val_prob, test_prob):
    '''
    Returns 0 if sample should be in train, 1 for val, and 2 for test
    '''
    pivot_val = random.uniform(0,1)

    if(pivot_val < train_prob):
        return 0
    elif(pivot_val >= train_prob and pivot_val < (train_prob+val_prob)):
        return 1
    else:
        return 2

def get_num_samples(sample_dir):
    return len(glob.glob1(sample_dir, "*.wav"))


