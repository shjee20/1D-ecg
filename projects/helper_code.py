#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

from operator import imod
import os, numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from scipy.signal import filtfilt, iirnotch, freqz, butter
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from sklearn.model_selection import StratifiedKFold
import ast 
import pickle
from scipy import signal
import glob
import pywt
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# (Re)sort leads using the standard order of leads for the standard twelve-lead ECG.
def sort_leads(leads):
    x = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    leads = sorted(leads, key=lambda lead: (x.index(lead) if lead in x else len(x) + leads.index(lead)))
    return tuple(leads)

def find_challenge_files(base_path):
    header_files = []
    recording_files = []

    # 첫 번째 수준의 하위 디렉토리 탐색 (예: chapman_shaoxing, cpst_2018 등)
    for subdir, dirs, _ in os.walk(base_path):
        for dir in dirs:
            # 'g'로 시작하는 두 번째 수준의 하위 디렉토리 탐색 (예: g1, g2 등)
            g_dirs_path = os.path.join(subdir, dir, "g*")
            g_dirs = glob.glob(g_dirs_path)

            for g_dir in g_dirs:
                # 'g' 디렉토리 내의 모든 .hea 파일 찾기
                dir_hea_files = glob.glob(os.path.join(g_dir, "*.hea"))
                # 'g' 디렉토리 내의 모든 .mat 파일 찾기
                dir_mat_files = glob.glob(os.path.join(g_dir, "*.mat"))

                # 찾은 .hea 파일들을 리스트에 추가
                for hea_file in dir_hea_files:
                    if os.path.isfile(hea_file):
                        header_files.append(hea_file)
                
                # 찾은 .mat 파일들을 리스트에 추가
                for mat_file in dir_mat_files:
                    if os.path.isfile(mat_file):
                        recording_files.append(mat_file)

    return header_files, recording_files


# Load header file as a string.
def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording

# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording

# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                recording_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return recording_id

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('#Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get number of leads from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_leads = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_leads

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[3])
            except:
                pass
        else:
            break
    return num_samples

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split('/')[0])
                except:
                    pass
        else:
            break
    return adc_gains

# Get baselines from header.
def get_baselines(header, leads):
    baselines = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    baselines[j] = float(entries[4].split('/')[0])
                except:
                    pass
        else:
            break
    return baselines

# Get labels from header.
def get_labels(header):
    labels = list()
    scored_labels = np.asarray(pd.read_csv("dx_mapping_scored.csv").iloc[:,1], dtype="str")
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    if any(j == entry for j in scored_labels):
                        labels.append(entry.strip())
            except:
                pass
    return labels

# Save outputs from model.
def save_outputs(output_file, recording_id, classes, labels, probabilities):
    # Format the model outputs.
    recording_string = '#{}'.format(recording_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = recording_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Save the model outputs.
    with open(output_file, 'w') as f:
        f.write(output_string)

# Load outputs from model.
def load_outputs(output_file):
    with open(output_file, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                recording_id = l[1:] if len(l)>1 else None
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(entry.strip() for entry in l.split(','))
            elif i==3:
                probabilities = tuple(float(entry) if is_finite_number(entry) else float('nan') for entry in l.split(','))
            else:
                break
    return recording_id, classes, labels, probabilities



def generator_to_list(generator, steps):
    data_list = []
    for _ in range(steps):
        data = next(generator)
        data = np.array(data)
        data_list.append(data)
    return data_list


def abbreviation(snomed_classes):
    SNOMED_scored = pd.read_csv("./dx_mapping_scored.csv", sep=",")
    snomed_abbr = []
    for j in range(len(snomed_classes)):
        for i in range(len(SNOMED_scored.iloc[:,1])):
            if (str(SNOMED_scored.iloc[:,1][i]) == snomed_classes[j]):
                snomed_abbr.append(SNOMED_scored.iloc[:,0][i])
                load_challenge
    snomed_abbr = np.asarray(snomed_abbr)
    return snomed_abbr

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def load_pkl_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def pan_tompkins(data, fs):
    lowcut = 5.0
    highcut = 15.0
    filter_order = 2
    nyquist_freq = 0.5 * fs

    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)

    diff_y = np.ediff1d(y)
    squared_diff_y=diff_y**2
    integrated_squared_diff_y =np.convolve(squared_diff_y,np.ones(5))

    max_h = integrated_squared_diff_y.max()

    peaks=find_peaks(integrated_squared_diff_y,height=max_h/2, distance=fs/3)

    if len(peaks[0]) > 1:
        hr = np.nanmean(60 /(np.diff(peaks[0])/fs)).mean()
    else:
        hr = np.nan 
    return hr

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def modify_path(path):
    # 경로를 \ 기준으로 분할
    parts = path.split('\\')
    # 뒤에서 두 번째 요소 제거
    modified_parts = parts[:-2] + parts[-1:]
    # 다시 경로를 조합
    new_path = '\\'.join(modified_parts)
    # 파일 확장자를 .mat에서 .pkl로 변경
    new_path = new_path.replace('.mat', '.pkl')
    return new_path

def calc_hr(ecg_filenames):
    heart_rate = np.zeros(len(ecg_filenames))
    for i,j in enumerate(ecg_filenames):
        data , head = load_challenge_data(j)
        
        heart_rate[i] = pan_tompkins(data,int(head[0].split(" ")[2]))
    heart_rate[np.where(np.isnan(heart_rate))[0]] = np.nanmean(heart_rate)
    return heart_rate

def calc_hr_predict(data):
    
    record , header = load_challenge_data(data)

    # 첫 번째 줄에서 샘플링 레이트 추출
    first_line = header[0]
    parts = first_line.split()
    sample_rate = int(parts[2])  # 예시에서는 '500'

    # pan_tompkins 함수를 사용하여 심박수 계산
    heart_rate = pan_tompkins(record, sample_rate)

    # 심박수 값이 유효하지 않은 경우 기본값으로 대체
    if np.isnan(heart_rate):
        heart_rate = 80

    return heart_rate




def batch_generator(batch_size, gen_x, gen_y, num_leads, num_classes): 
    #np.random.shuffle(order_array)
    batch_features = np.zeros((batch_size,1250, num_leads))
    batch_labels = np.zeros((batch_size,num_classes))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
        yield batch_features, batch_labels    

def generate_y(y_train):
    while True:
        for i in y_train:
            yield i

def standardize_signal(signal):
    """Z-transform을 적용하고 scale_factor를 곱하여 원래 크기 유지"""
    mean = np.mean(signal)
    std = np.std(signal)
    standardized_signal = (signal - mean) / std #if std != 0 else signal  # 표준화 수행
    return standardized_signal  # 신호 크기 조정

def modify_path(path):
    # 경로를 \ 기준으로 분할
    parts = path.split('\\')
    # 뒤에서 두 번째 요소 제거
    modified_parts = parts[:-2] + parts[-1:]
    # 다시 경로를 조합
    new_path = '\\'.join(modified_parts)
    # 파일 확장자를 .mat에서 .pkl로 변경
    new_path = new_path.replace('.mat', '.pkl')
    return new_path

def load_pkl_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist 주파수 (샘플링 주파수의 절반)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Butterworth 밴드패스 필터 생성
    filtered_signal = filtfilt(b, a, signal)  # 필터 적용 (위상 왜곡 방지를 위해 filtfilt 사용)
    return filtered_signal

def calculate_original_max_values(original_data, num_leads):
    max_values = np.max(np.abs(original_data[:num_leads, :]), axis=1)
    return max_values

def adjust_sampling_rate(data, header_data, original_max_values, target_frequency=500):
    original_frequency = int(header_data[0].split(" ")[2])
    # 정규화 복원
    for i in range(data.shape[0]):
        data[i] = data[i] * original_max_values[i]
    if original_frequency != target_frequency:
        new_length = int((data.shape[1] / original_frequency) * target_frequency)
        data_new = np.ones([data.shape[0], new_length])
        for i in range(data.shape[0]):
            data_new[i] = signal.resample(data[i], new_length)
        data = data_new
    return data


'''Use in traning'''

def extract_dx_from_hea(file_path):
    """
    Reads a .hea file and extracts the diagnosis code.
    
    Parameters:
    file_path (str): Path to the .hea file.
    
    Returns:
    str: The extracted diagnosis code.
    """
    dx_code = ''
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('# Dx:'):
                    dx_code = line.replace('# Dx:', '').strip()
                    break
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return dx_code


def extract_dx(hea):
    """
    Reads a .hea file and extracts the diagnosis code.
    
    Parameters:
    file_path (str): Path to the .hea file.
    
    Returns:
    str: The extracted diagnosis code.
    """
    dx_code = ''
    try:
        
        for line in hea:
            if line.startswith('# Dx:'):
                dx_code = line.replace('# Dx:', '').strip()
                break
    except FileNotFoundError:
        print("fail")
    return dx_code


def check_and_print_nonexistent_files(file_paths):
    for file_path in file_paths:
        if file_path.endswith('.mat'):
            hea_path = file_path.replace('.mat', '.hea')
            if not (os.path.isfile(file_path) and os.path.isfile(hea_path)):
                print(f"Both files do not exist: {file_path} and {hea_path}")



def stratified_sample(files, labels, hr_data, n_samples=100):
    skf = StratifiedKFold(n_splits=(len(labels) // n_samples), shuffle=True, random_state=42)
    for train_index, _ in skf.split(files, labels.argmax(axis=1)):
        return np.array(files)[train_index], labels[train_index], np.array(hr_data)[train_index]



def perform_discrete_wavelet_transform(ecg_signal, lead_index):
    # Perform discrete wavelet decomposition at level 8
    coeffs = pywt.wavedec(ecg_signal, 'db10', level=10)
    
    # Remove the approximation coefficients (C8) by zeroing them out
    coeffs[0] = np.zeros_like(coeffs[0])
    
    # Reconstruct the signal using the modified coefficients
    filtered_signal = pywt.waverec(coeffs, 'db10')
    
    # Adjust the reconstructed signal length to match the input length
    original_length = len(ecg_signal)
    filtered_length = len(filtered_signal)
    
    if filtered_length > original_length:
        # If the output is longer, trim the extra elements
        filtered_signal = filtered_signal[:original_length]
    elif filtered_length < original_length:
        # If the output is shorter, pad with zeros at the end
        pad_width = original_length - filtered_length
        filtered_signal = np.pad(filtered_signal, (0, pad_width), mode='constant')
    
    return filtered_signal

def remove_same_order(seq):
    """입력된 리스트에서 중복 항목을 제거하고, 원래 순서를 유지하는 함수."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result



# class ECGDataset(Dataset):
#     def __init__(self, gen_x, gen_y, num_leads, num_classes):
 
#         self.gen_x = gen_x
#         self.gen_y = gen_y
#         self.num_leads = num_leads
#         self.num_classes = num_classes

#     def __len__(self):
#         return 10000  # 가상의 데이터 크기 (제너레이터 기반이므로 고정값 설정)

#     def __getitem__(self, idx):
#         """데이터 1개 샘플을 가져오는 메서드"""
#         batch_features = next(self.gen_x)  # ECG 데이터 가져오기
#         batch_labels = next(self.gen_y)  # 라벨 가져오기

#         batch_features = torch.tensor(batch_features, dtype=torch.float32)  # PyTorch Tensor 변환
#         batch_labels = torch.tensor(batch_labels, dtype=torch.float32)

#         return batch_features, batch_labels

# class F1Score(Metric):
#     def __init__(self, name="f1_score", **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.tp = self.add_weight(name="tp", initializer="zeros")
#         self.fp = self.add_weight(name="fp", initializer="zeros")
#         self.fn = self.add_weight(name="fn", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.round(y_pred)  # 0.5 기준으로 반올림하여 이진 예측 변환

#         tp = tf.reduce_sum(y_true * y_pred)
#         fp = tf.reduce_sum((1 - y_true) * y_pred)
#         fn = tf.reduce_sum(y_true * (1 - y_pred))

#         self.tp.assign_add(tp)
#         self.fp.assign_add(fp)
#         self.fn.assign_add(fn)

#     def result(self):
#         precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
#         recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
#         return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

#     def reset_states(self):
#         self.tp.assign(0)
#         self.fp.assign(0)
#         self.fn.assign(0)


class NaNLossRecoveryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(NaNLossRecoveryCallback, self).__init__()
        self.last_good_weights = None
        self.last_good_epoch = 0

    def on_train_begin(self, logs=None):
        self.last_good_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is None:
            return

        if np.isnan(loss) or np.isinf(loss):
            print(f"\n [Epoch {epoch + 1}] Train loss is NaN/Inf → 복원 중 (이전 epoch {self.last_good_epoch + 1})")
            self.model.set_weights(self.last_good_weights)
            drop_rate = 0.8
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            min_lr = 1e-5
            new_lr = max(lr * drop_rate, min_lr)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"Learning rate reduced: {old_lr:.5e} → {new_lr:.5e}")


        else:
            self.last_good_weights = self.model.get_weights()
            self.last_good_epoch = epoch


def step_decay_schedule(epoch, lr, *, drop_rate=0.8, step_size=10, min_lr=1e-5):
    """
    • epoch : 0-based epoch index (콜백에서 자동 전달)
    • lr    : 현재 학습률 (전 epoch에서 사용된 값)
    반환값  : 이번 epoch 에 사용할 학습률
    """
    # epoch가 step_size 배수일 때만 감소
    if epoch and epoch % step_size == 0:
        lr = max(lr * drop_rate, min_lr)   # 하한선(min_lr) 유지
    return lr


class AsymmetricLoss(tf.keras.losses.Loss):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, from_logits=True, name="asymmetric_loss"):
        super().__init__(name=name)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # Apply sigmoid if from_logits is True
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        # Positive & negative probabilities
        xs_pos = y_pred
        xs_neg = 1.0 - y_pred

        # Asymmetric Clipping (to prevent log(0))
        if self.clip is not None and self.clip > 0:
            xs_neg = tf.clip_by_value(xs_neg + self.clip, 0.0, 1.0)

        # Log components
        loss_pos = y_true * tf.math.log(tf.clip_by_value(xs_pos, self.eps, 1.0))
        loss_neg = (1.0 - y_true) * tf.math.log(tf.clip_by_value(xs_neg, self.eps, 1.0))

        # Combine
        loss = loss_pos + loss_neg

        # Focal Modulation (no gradient)
        pt = y_true * xs_pos + (1.0 - y_true) * xs_neg
        one_sided_gamma = self.gamma_pos * y_true + self.gamma_neg * (1.0 - y_true)
        focal_weight = tf.stop_gradient(tf.pow(1.0 - pt, one_sided_gamma))  # detach gradient
        loss *= focal_weight

        return -tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # batch-wise mean

class TrainSigmoidLogger(Callback):
    def __init__(self, x_train_sample, save_dir=None):
        super().__init__()
        self.x_train_sample = x_train_sample  # (N, 5000, 12)
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_train_sample, verbose=0)

        # 1️⃣ 저장
        if self.save_dir:
            np.save(f"{self.save_dir}/sigmoid_train_epoch{epoch}.npy", y_pred)

        # 2️⃣ 시각화
        plt.figure(figsize=(6, 4))
        plt.hist(y_pred.flatten(), bins=50, color='steelblue')
        plt.title(f"Train Sigmoid Output - Epoch {epoch}")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.grid(True)
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/sigmoid_hist_epoch{epoch}.png")
            plt.close()
        else:
            plt.show()

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='macro', threshold=0.5, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold

        self.tp = self.add_weight(name='true_positives', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='false_positives', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='false_negatives', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_pred * y_true, axis=0))
        self.fp.assign_add(tf.reduce_sum(y_pred * (1 - y_true), axis=0))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred) * y_true, axis=0))

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())

        if self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'micro':
            total_tp = tf.reduce_sum(self.tp)
            total_fp = tf.reduce_sum(self.fp)
            total_fn = tf.reduce_sum(self.fn)
            precision = total_tp / (total_tp + total_fp + K.epsilon())
            recall = total_tp / (total_tp + total_fn + K.epsilon())
            return 2 * precision * recall / (precision + recall + K.epsilon())
        else:
            return f1  # per class

    def reset_states(self):
        for var in self.variables:
            var.assign(tf.zeros_like(var))


# class ROCThresholdCSVCallback(tf.keras.callbacks.Callback):
#     def __init__(self, x_val, y_val, f1_metric,
#                  batch_size=16,
#                  save_dir='th_log',
#                  prefix='raw_12lead',
#                  threshold_candidates=None):
#         super().__init__()
#         self.x_val = x_val
#         self.y_val = y_val
#         self.f1_metric = f1_metric
#         self.batch_size = batch_size
#         self.save_dir = save_dir
#         self.filename = f"threshold_{prefix}.csv"
#         self.save_path = os.path.join(save_dir, self.filename)
#         self.history = []

#         # threshold 탐색 범위 설정 (coarse sweep)
#         if threshold_candidates is None:
#             self.threshold_candidates = np.arange(0.1, 0.9, 0.05)
#         else:
#             self.threshold_candidates = threshold_candidates

#         os.makedirs(self.save_dir, exist_ok=True)

#     def on_epoch_end(self, epoch, logs=None):
#         print(f"[INFO] ROCThresholdCSVCallback: Epoch {epoch} 시작")

#         try:
#             # GPU 사용 시 메모리 폭발 방지용 → CPU에서 실행 권장
#             with tf.device("/CPU:0"):
#                 y_pred_probs = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
#             print("[INFO] 예측 완료")
#         except Exception as e:
#             print("[ERROR] Prediction failed:", e)
#             return

        try:
            num_classes = y_pred_probs.shape[1]
            best_thresholds = []

            for i in range(num_classes):
                y_true = self.y_val[:, i]
                y_prob = y_pred_probs[:, i]

                # 모두 0 or 모두 1이면 threshold 튜닝 무의미 → 0.5 고정
                if np.all(y_true == 0) or np.all(y_true == 1):
                    best_thresholds.append(0.5)
                    continue

                best_f1 = -1
                best_th = 0.5  # fallback
                for th in self.threshold_candidates:
                    y_pred_bin = (y_prob > th).astype(int)
                    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_th = th

                best_thresholds.append(best_th)

            best_thresholds = np.round(best_thresholds, 4)
            self.history.append(best_thresholds)

            df = pd.DataFrame(self.history, columns=[f'Class_{i}' for i in range(num_classes)])
            df.index.name = 'Epoch'
            df.to_csv(self.save_path)
            print(f"[INFO] Thresholds saved to {self.save_path}")

            self.f1_metric.set_thresholds(best_thresholds)

        except Exception as e:
            print(f"[ERROR] Threshold computation failed: {str(e)}")


class DynamicF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='macro', threshold=0.2, name='dynamic_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self._build_metric()

    def _build_metric(self):
        self._f1 = F1Score(num_classes=self.num_classes, average=self.average, threshold=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return self._f1.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self._f1.result()

    def reset_states(self):
        return self._f1.reset_states()

    def set_threshold(self, threshold):
        self.threshold = threshold
        self._build_metric()

def generator_to_array(gen_x, gen_y, total_len, num_leads, num_classes):
    """
    Generator를 정적인 배열로 변환 (validation용)
    """
    x_array = np.zeros((total_len, 5000, num_leads), dtype=np.float32)
    y_array = np.zeros((total_len, num_classes), dtype=np.float32)

    for i in range(total_len):
        x_array[i] = next(gen_x)
        y_array[i] = next(gen_y)

    return x_array, y_array

class CustomF1WithClassThresholds(tf.keras.metrics.Metric):
    def __init__(self, num_classes, thresholds=None, average='macro', name='dynamic_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average

        # 임계값은 float32로 고정
        init_thr = tf.convert_to_tensor(
            thresholds if thresholds is not None else [0.5] * num_classes,
            dtype=tf.float32
        )
        self.thresholds = tf.Variable(init_thr, trainable=False, dtype=tf.float32)

        # 누적 통계는 float32로 고정
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 혼합정밀 대비: 비교 전 일괄 float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # (배치, C) 비교 위해 thresholds를 (1, C)로 브로드캐스트
        thr = tf.reshape(tf.cast(self.thresholds, tf.float32), [1, -1])

        y_pred_bin = tf.cast(y_pred > thr, tf.float32)

        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            # sw shape을 (batch, 1)로 브로드캐스트
            if tf.rank(sw) == 1:
                sw = tf.reshape(sw, [-1, 1])
            y_pred_bin = y_pred_bin * sw
            y_true     = y_true * sw

        self.tp.assign_add(tf.reduce_sum(y_pred_bin * y_true, axis=0))
        self.fp.assign_add(tf.reduce_sum(y_pred_bin * (1 - y_true), axis=0))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred_bin) * y_true, axis=0))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1) if self.average == 'macro' else f1

    def reset_states(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

    def set_thresholds(self, new_thresholds):
        self.thresholds.assign(tf.cast(new_thresholds, tf.float32))
