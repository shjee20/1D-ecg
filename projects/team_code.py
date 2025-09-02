
 
### 설치해야할 주요 라이브러리는 다음과 같습니다.

# !pip install numpy
# !pip install pandas
# !pip install scikit-learn
# !pip install seaborn
# !pip install matplotlib
# !pip install tensorflow
# !pip install tensorflow-addons
# !pip install scipy
# !pip install joblib
# !pip install ast
# !pip install pickle

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
from model_type.ResNet import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import pandas as pd
from scipy import signal
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger
from sklearn.model_selection import train_test_split
import ast 
import pickle
import matplotlib.pyplot as plt
import IPython.display as display
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tqdm import tqdm
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, LambdaCallback
import tensorflow.keras.backend as K
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score


# 모델 컴파일 시 추가 가능
metrics=['binary_accuracy']
# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

other_diag = ["bundle branch block","bradycardia","1st degree av block", "incomplete right bundle branch block", "left axis deviation", "left anterior fascicular block", "left bundle branch block", "low qrs voltages",
        "nonspecific intraventricular conduction disorder", "poor R wave Progression", "prolonged pr interval", "prolonged qt interval", "qwave abnormal", "right axis deviation", "right bundle branch block", "t wave abnormal", "t wave inversion"]




################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.





def run_model(model, recordings, num_leads, f_num, best_th):
    """
    model: 학습된 keras 모델
    recordings: 테스트할 ECG 데이터 리스트
    num_leads: 사용할 리드 개수
    f_num: 전처리 방식 인덱스 (0=raw, 1=dwt, 2=pan-tomkins)
    best_th: 클래스별 threshold 리스트, 길이는 모델의 출력 차원과 동일
    """
    th_array = np.array(best_th).reshape(1, -1)  

    all_probabilities = []
    all_binary_predictions = []
    
    # 전처리 방식 선택
    if f_num == 0:
        ecg_generator = generate_X_rawecg(recordings, num_leads)
    elif f_num == 1:
        ecg_generator = generate_X_dwt(recordings, num_leads)
    elif f_num == 2:
        ecg_generator = generate_X_pan_tomkins(recordings, num_leads)
    else:
        raise ValueError(f"Unknown f_num: {f_num}")

    total = len(recordings)
    for idx, ecg_data in enumerate(tqdm(ecg_generator, total=total, desc="Predicting")):
 
        if idx >= total:
            break

        x = np.expand_dims(ecg_data, axis=0)  # shape = (1, seq_len, num_leads)
        probs = model.predict(x, verbose=0)   # shape = (1, n_classes)

        binary = (probs > th_array).astype(int)

        all_probabilities.append(probs)
        all_binary_predictions.append(binary)

    return all_binary_predictions, all_probabilities

def run_model_2(model, data, num_leads, best_th):
    """
    model: 학습된 keras 모델
    recordings: 테스트할 ECG 데이터 리스트
    num_leads: 사용할 리드 개수
    f_num: 전처리 방식 인덱스 (0=raw, 1=dwt, 2=pan-tomkins)
    best_th: 클래스별 threshold 리스트, 길이는 모델의 출력 차원과 동일
    """
    th_array = np.array(best_th).reshape(1, -1)  

    all_probabilities = []
    all_binary_predictions = []
    ecg_generator = generate_X(data)
    
    # # 전처리 방식 선택
    # if f_num == 0:
    #     ecg_generator = generate_X_rawecg(recordings, num_leads)
    # elif f_num == 1:
    #     ecg_generator = generate_X_dwt(recordings, num_leads)
    # elif f_num == 2:
    #     ecg_generator = generate_X_pan_tomkins(recordings, num_leads)
    # else:
    #     raise ValueError(f"Unknown f_num: {f_num}")

    total = len(data)
    for idx, ecg_data in enumerate(tqdm(ecg_generator, total=total, desc="Predicting")):
 
        if idx >= total:
            break

        x = np.expand_dims(ecg_data, axis=0)  # shape = (1, seq_len, num_leads)
        probs = model.predict(x, verbose=0)   # shape = (1, n_classes)

        binary = (probs > th_array).astype(int)

        all_probabilities.append(probs)
        all_binary_predictions.append(binary)

    return all_binary_predictions, all_probabilities

def load_model(model_directory):
    # 모델 로드
    model = tf.keras.models.load_model(model_directory, compile=False)
    return model

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_{}_leads.h5'.format(len(sorted_leads))

################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.

#-----------------------------------------------------------#
#                                                           #
#                    My functions                           #
#                                                           #
#-----------------------------------------------------------#
def print_label_counts(name, y_subset, abbreviation):
    class_counts = np.sum(y_subset, axis=0)  # 각 열(클래스)별 1의 개수 계산
    print(f"=== {name} ===")
    for i, count in enumerate(class_counts):
        print(f"{abbreviation[i]}: {int(count)}")
    print()

def save_label_counts_to_excel(y_train, y_val, y_test, abbreviation, output_path="label_distribution.xlsx"):
    # 클래스별 1의 개수 계산
    train_counts = np.sum(y_train, axis=0)
    val_counts = np.sum(y_val, axis=0)
    test_counts = np.sum(y_test, axis=0)

    # 데이터프레임 구성
    df = pd.DataFrame({
        'Abbreviation': abbreviation,
        'Train': train_counts.astype(int),
        'Validation': val_counts.astype(int),
        'Test': test_counts.astype(int)
    })

    # 엑셀 저장
    df.to_excel(output_path, index=False)
    print(f"[INFO] 클래스별 개수 분포를 {output_path}로 저장했습니다.")

def load_data(X_train_file, num_leads=12, target_fs=500, target_duration_sec=10):
    print("load_start")
    X_all = []
    y_all = []

    for h in X_train_file:
        data, header_data = load_challenge_data(h)

        fs = int(header_data[0].split(" ")[2])          # sampling frequency
        sample_num = int(header_data[0].split(" ")[3])  # number of samples
        duration = sample_num / fs                      # seconds
        target_len = int(target_fs * target_duration_sec)  # 5000

        # 리샘플링
        if fs != target_fs:
            new_len = int(duration * target_fs)
            data_resampled = np.zeros((num_leads, new_len))
            for i, lead in enumerate(data):
                data_resampled[i] = signal.resample(lead, new_len)
            data = data_resampled  # shape: (num_leads, new_len)

        # 자르거나 패딩해서 정확히 5000 길이 맞추기
        data = pad_sequences(data, maxlen=target_len, truncating='post', padding='post')

        # 차원 유지: (num_leads, 5000) → generate_X_rawecg와 호환됨
        X_all.append(data)
        y_all.append(header_data)

    print("load_end")
    return X_all, y_all

def is_invalid_sample(data, threshold=8000):
    for lead in data:
        if np.isnan(lead).any():
            return True
        if np.std(lead) < 1e-3:
            return True
        if np.max(np.abs(lead)) > threshold:
            return True
    return False


def load_data_2(X_file, num_leads=12, target_fs=500, target_duration_sec=10):
    print("load_start")
    X_all = []
    y_all = []
    mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for h in X_file:
        data, header_data = load_challenge_data(h)

        if is_invalid_sample(data):
            continue  # 이 샘플은 skip (혹은 log 기록)

        # 헤더 예외 처리
        if not header_data or len(header_data) == 0:
            print(f"[경고] 헤더가 없거나 비어 있음: {h}")
            continue

        try:
            fs = int(header_data[0].split(" ")[2])         
            sample_num = int(header_data[0].split(" ")[3])  
        except Exception as e:
            print(f"[에러] 헤더 파싱 실패: {h} → {e}")
            continue

        duration = sample_num / fs                        
        target_len = int(target_fs * target_duration_sec)  

        # 리샘플링
        if fs != target_fs:
            new_len = int(duration * target_fs)
            data_resampled = np.zeros((num_leads, new_len))
            for i, lead in enumerate(data):
                data_resampled[i] = signal.resample(lead, new_len)
            data = data_resampled  # shape: (num_leads, new_len)

        # 자르거나 패딩해서 정확히 5000 길이 맞추기
        data = pad_sequences(data, maxlen=target_len, truncating='post', padding='post')

        # 차원 유지: (num_leads, 5000) → generate_X_rawecg와 호환됨
        X_all.append(data)
        y_all.append(header_data)

    print(len(y_all))
    df_code = pd.read_csv('SNOMEDCTCode_Mapping.csv')
    remove_list = ["BBB", "LPR", "Brady"]
    df_code = df_code[~df_code["Abbreviation"].isin(remove_list)]
    class_list = df_code['SNOMEDCTCode'].to_list()
    classes = [item.split(',') if ',' in item else [item] for item in class_list]


    y_dx = [extract_dx(hea) for hea in y_all]
    y_labels, valid_indices, class_order = process_data(y_dx, classes)

    abbreviation_sorted = df_code['Abbreviation'].tolist()
    abbreviation_sorted = [abbreviation_sorted[i] for i in class_order]


    X_all = np.array(X_all)[valid_indices]
    
    folds = []
    # for _, test_index in mskf.split(X_files, y_data):
    for jj, index in mskf.split(X_all, y_labels):  #뒤에 X_files는 더미
        folds.append(index)

    test_index = folds[0]
    val_index = folds[1]
    train_index = np.concatenate([folds[i] for i in range(2, 10)])

    # X와 y 모두 동일하게 인덱스 적용
    X_all = np.array(X_all)
    X_train = X_all[train_index]
    X_val   = X_all[val_index]
    X_test  = X_all[test_index]

    y_train = y_labels[train_index]
    y_val   = y_labels[val_index]
    y_test  = y_labels[test_index]

    print_label_counts("Train", y_train, abbreviation_sorted)
    print_label_counts("Validation", y_val, abbreviation_sorted)
    print_label_counts("Test", y_test, abbreviation_sorted)
    save_label_counts_to_excel(y_train, y_val, y_test, abbreviation_sorted)

    print("load_end")
    return X_train, y_train, X_val, y_val, X_test, y_test, abbreviation_sorted

def process_data(dx_list, classes):
    one_hot_list = []
    valid_indices = []

    for i, dx_code in enumerate(dx_list):
        dx_codes = dx_code.split(',')
        one_hot = [0] * len(classes)
        for idx, class_group in enumerate(classes):
            if any(code in dx_codes for code in class_group):
                one_hot[idx] = 1

        if any(one_hot):
            one_hot_list.append(one_hot)
            valid_indices.append(i)

    y_array = np.array(one_hot_list)  # (N_valid, num_classes)

    # 클래스별 등장 횟수 계산 및 정렬
    class_counts = np.sum(y_array, axis=0)
    class_order = np.argsort(-class_counts)

    y_array_sorted = y_array[:, class_order]

    return y_array_sorted, valid_indices, class_order



def train_model_2(train_data, train_labels, val_data, val_labels, num_leads, batch_size, lr, epochs, signal_len, model_name, model_directory, f_num):

    model, f1_metric = encoder_resnet_1d((signal_len, num_leads), 23, lr)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_directory, f"{model_name}_e{{epoch:02d}}_valf1{{val_dynamic_f1:.2f}}.h5"),
        monitor="val_dynamic_f1",
        verbose=1,
        save_best_only=False,
        mode="max"
    )


    print('Train Start...')
    
    
    # 로그 저장 경로 설정
    log_dir = f"logs/fit/{model_name}_bs{batch_size}_lr{lr}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
    nan_recovery_callback = NaNLossRecoveryCallback()

    x_train_sample, _ = next(batch_generator_1_train(
    batch_size=batch_size,
    gen_x=generate_X_rawecg(train_data, num_leads),
    gen_y=generate_y(train_labels),
    num_leads=num_leads,
    num_classes=23
    ))

    sigmoid_logger = TrainSigmoidLogger(x_train_sample=x_train_sample, save_dir="./sigmoid_logs")
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule, verbose=1)

    Early_Stopping = EarlyStopping(
        monitor='val_dynamic_f1', 
        patience=20,             
        mode='max'               
    )

    '''f_num = 0 은 raw signal임 '''
    
    if f_num == 0:

        gen_x = generate_X_rawecg(val_data, num_leads)
        gen_y = generate_y(val_labels)

        x_val, y_val = generator_to_array(gen_x, gen_y, total_len=len(val_data), num_leads=num_leads, num_classes=23)

        # roc_callback = ROCThresholdCSVCallback(
        #     x_val=x_val,
        #     y_val=y_val,
        #     f1_metric=f1_metric,
        #     prefix=f"PT_{num_leads}lead"  # 예: raw_12lead
        # )
        callbacks=[checkpoint_callback, tensorboard_callback, nan_recovery_callback, Early_Stopping, sigmoid_logger]

        
        history = model.fit(
            x=batch_generator_1(
                batch_size=batch_size, 
                gen_x=generate_X_rawecg(train_data, num_leads),  
                gen_y=generate_y(train_labels), 
                num_leads=num_leads, 
                num_classes=23
            ),
            epochs=epochs, 
            steps_per_epoch=(len(train_data) // batch_size),
            validation_data=batch_generator_1(
                batch_size=batch_size, 
                gen_x=generate_X_rawecg(val_data, num_leads),  
                gen_y=generate_y(val_labels), 
                num_leads=num_leads, 
                num_classes=23
            ), 
            validation_steps=(len(val_labels) // batch_size),
            callbacks=callbacks
        )
        gen = batch_generator_1(
        batch_size=len(val_data), 
        gen_x=generate_X_rawecg(val_data, num_leads),  
        gen_y=generate_y(val_labels), 
        num_leads=num_leads, 
        num_classes=23
        )

        '''f_num = 1 은 dwt signal임 '''

    elif f_num == 1:
 
        gen_x = generate_X_dwt(val_data, num_leads)
        gen_y = generate_y(val_labels)
        x_val, y_val = generator_to_array(gen_x, gen_y, total_len=len(val_data), num_leads=num_leads, num_classes=23)
        # roc_callback = ROCThresholdCSVCallback(
        #     x_val=x_val,
        #     y_val=y_val,
        #     f1_metric=f1_metric,
        #     prefix=f"PT_{num_leads}lead"  # 예: raw_12lead
        # )


        callbacks=[checkpoint_callback, tensorboard_callback, nan_recovery_callback, Early_Stopping, sigmoid_logger]
        history = model.fit(
            x=batch_generator_1(
                batch_size=batch_size, 
                gen_x=generate_X_dwt(train_data, num_leads),  
                gen_y=generate_y(train_labels), 
                num_leads=num_leads, 
                num_classes=23
            ),
            epochs=epochs, 
            steps_per_epoch=(len(train_data) // batch_size),
            validation_data=batch_generator_1(
                batch_size=batch_size, 
                gen_x=generate_X_dwt(val_data, num_leads),  
                gen_y=generate_y(val_labels), 
                num_leads=num_leads, 
                num_classes=23
            ),
            validation_steps=(len(val_labels) // batch_size),
            callbacks=callbacks
        )

        '''f_num = 2 은 PT signal임 '''

    elif f_num == 2:


        gen_x = generate_X_pan_tomkins(val_data, num_leads)
        gen_y = generate_y(val_labels)

        x_val, y_val = generator_to_array(gen_x, gen_y, total_len=len(val_data), num_leads=num_leads, num_classes=23)

        # roc_callback = ROCThresholdCSVCallback(
        #     x_val=x_val,
        #     y_val=y_val,
        #     f1_metric=f1_metric,
        #     prefix=f"PT_{num_leads}lead"  # 예: raw_12lead
        # )
        callbacks=[checkpoint_callback, tensorboard_callback, nan_recovery_callback, Early_Stopping, sigmoid_logger]

        history = model.fit(
            x=batch_generator_1(
                batch_size=batch_size, 
                gen_x=generate_X_pan_tomkins(train_data, num_leads), 
                gen_y=generate_y(train_labels), 
                num_leads=num_leads, 
                num_classes=23
            ),
            epochs=epochs, 
            steps_per_epoch=(len(train_data) // batch_size),
            validation_data=batch_generator_1(
                batch_size=batch_size, 
                gen_x= generate_X_pan_tomkins(val_data, num_leads),  
                gen_y=generate_y(val_labels), 
                num_leads=num_leads, 
                num_classes=23
            ),
            validation_steps=(len(val_labels) // batch_size),
            callbacks=callbacks
        )


def train_model_3(train_data, train_labels, val_data, val_labels, num_leads, batch_size, lr, epochs, signal_len, model_name, model_directory, f_num):

    model, f1_metric = encoder_resnet_1d((signal_len, num_leads), 23, lr)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_directory, f"{model_name}_e{{epoch:02d}}_valf1{{val_dynamic_f1:.2f}}.h5"),
        monitor="val_dynamic_f1",
        verbose=1,
        save_best_only=False,
        mode="max"
    )


    print('Train Start...')
    
    
    # 로그 저장 경로 설정
    log_dir = f"logs/fit/{model_name}_bs{batch_size}_lr{lr}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
    nan_recovery_callback = NaNLossRecoveryCallback()

    # x_train_sample, _ = next(batch_generator_1_train(
    # batch_size=batch_size,
    # gen_x=generate_X_rawecg(train_data, num_leads),
    # gen_y=generate_y(train_labels),
    # num_leads=num_leads,
    # num_classes=23
    # ))

    # sigmoid_logger = TrainSigmoidLogger(x_train_sample=x_train_sample, save_dir="./sigmoid_logs")
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule, verbose=1)

    Early_Stopping = EarlyStopping(
        monitor='val_dynamic_f1', 
        patience=20,             
        mode='max'               
    )
    
    # roc_callback = ROCThresholdCSVCallback(
    #     x_val=x_val,
    #     y_val=y_val,
    #     f1_metric=f1_metric,
    #     prefix=f"PT_{num_leads}lead"  # 예: raw_12lead
    # )
    callbacks=[lr_scheduler, checkpoint_callback, tensorboard_callback, nan_recovery_callback, Early_Stopping]

    
    history = model.fit(
        x=batch_generator_1(
            batch_size=batch_size, 
            gen_x=generate_X(train_data),  
            gen_y=generate_y(train_labels), 
            num_leads=num_leads, 
            num_classes=23
        ),
        epochs=epochs, 
        steps_per_epoch=(len(train_data) // batch_size),
        validation_data=batch_generator_1(
            batch_size=batch_size, 
            gen_x=generate_X(val_data),  
            gen_y=generate_y(val_labels), 
            num_leads=num_leads, 
            num_classes=23
        ), 
        validation_steps=(len(val_labels) // batch_size),
        callbacks=callbacks
    )
    # gen = batch_generator_1(
    # batch_size=len(val_data), 
    # gen_x=generate_X(val_data),  
    # gen_y=generate_y(val_labels), 
    # num_leads=num_leads, 
    # num_classes=23
    # )


def generate_X_rawecg(X_data, num_leads):
    while True:
        for data in X_data:
            if num_leads == 12:
                data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
            elif num_leads == 6:
                data = data[[0,1,2,3,4,5]]
            elif num_leads == 4:
                data = data[[0,1,2,7]]
            elif num_leads == 3:
                data = data[[0,1,7]]
            elif num_leads == 2:
                data = data[[0,1]]
            elif num_leads == 1:
                data = data[[1]]

            # Apply standardization and bandpass filter on each lead
            for lead_index in range(data.shape[0]):
                data[lead_index, :] = bandpass_filter(data[lead_index, :], 0.05, 25, 500)
                data[lead_index, :] = standardize_signal(data[lead_index, :])


            # Reshape (swap axes) before yielding
            data = data.reshape(data.shape[1], data.shape[0])
            yield data

def generate_X_dwt(X_data, num_leads):
    while True:

        for data in X_data:
            if num_leads == 12:
                data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
            elif num_leads == 6:                                     
                data = data[[0,1,2,3,4,5]]
            elif num_leads == 4:
                data = data[[0,1,2,7]]
            elif num_leads == 3:
                data = data[[0,1,7]]
            elif num_leads == 2:
                data = data[[0,1]]
            elif num_leads == 1:
                data = data[[1]]

            #data = data + np.random.choice([0,0,0,np.random.rand(12,5000)*random.randint(0, 50)])

            for lead_index in range(data.shape[0]):  # 리드 개수만큼 반복 (행 기준)
              data[lead_index, :] = perform_discrete_wavelet_transform(data[lead_index, :], lead_index) 
              data[lead_index, :] = bandpass_filter(data[lead_index, :], 0.05, 25, 500)  # 밴드패스 필터 적용
              data[lead_index, :] = standardize_signal(data[lead_index, :]) 

            data = data.reshape(data.shape[1],data.shape[0])
            yield data
          
def generate_X_pan_tomkins(X_data, num_leads):
    while True:
        for data in X_data:

            if num_leads == 12:
                data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
            elif num_leads == 6:                                     
                data = data[[0,1,2,3,4,5]]
            elif num_leads == 4:
                data = data[[0,1,2,7]]
            elif num_leads == 3:
                data = data[[0,1,7]]
            elif num_leads == 2:
                data = data[[0,1]]
            elif num_leads == 1:
                data = data[[1]]
            # 각 리드별로 NaN 값 보간
            for i in range(data.shape[0]):
                lead_data = data[i, :]
                lead_series = pd.Series(lead_data)
                lead_series = lead_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                data[i, :] = lead_series.values
                data[i, :] = standardize_signal(data[i, :]) 

            data = data.reshape(data.shape[1], data.shape[0])

            yield data


def X_rawecg(X_data, num_leads):
    filtered_X_data = []

    for data in tqdm(X_data, desc="Preprocessing with bandpass and standardization"):
        # 선택된 리드 추출
        if num_leads == 12:
            data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
        elif num_leads == 6:
            data = data[[0,1,2,3,4,5]]
        elif num_leads == 4:
            data = data[[0,1,2,7]]
        elif num_leads == 3:
            data = data[[0,1,7]]
        elif num_leads == 2:
            data = data[[0,1]]
        elif num_leads == 1:
            data = data[[1]]

        # 리드별 전처리 (bandpass + standardize)
        for lead_index in range(data.shape[0]):
            data[lead_index, :] = bandpass_filter(data[lead_index, :], 0.05, 25, 500)
            data[lead_index, :] = standardize_signal(data[lead_index, :])

        # Shape 변경 및 리스트에 추가
        data = data.reshape(data.shape[1], data.shape[0])  # (5000, num_leads)
        filtered_X_data.append(data)

    return np.array(filtered_X_data)

def X_dwt(X_data, num_leads):
    filtered_X_data = []

    for data in tqdm(X_data, desc="Preprocessing with DWT + bandpass + standardization"):
        if num_leads == 12:
            data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
        elif num_leads == 6:                                     
            data = data[[0,1,2,3,4,5]]
        elif num_leads == 4:
            data = data[[0,1,2,7]]
        elif num_leads == 3:
            data = data[[0,1,7]]
        elif num_leads == 2:
            data = data[[0,1]]
        elif num_leads == 1:
            data = data[[1]]

        for lead_index in range(data.shape[0]):
            data[lead_index, :] = bandpass_filter(data[lead_index, :], 0.05, 25, 500)
            data[lead_index, :] = perform_discrete_wavelet_transform(data[lead_index, :], lead_index)
            data[lead_index, :] = standardize_signal(data[lead_index, :])

        data = data.reshape(data.shape[1], data.shape[0])
        filtered_X_data.append(data)

    return np.array(filtered_X_data)
          
def X_pan_tomkins(X_data, num_leads):
    filtered_X_data = []

    for data in tqdm(X_data, desc="Preprocessing with Pan-Tompkins + interpolate + standardize"):
        if num_leads == 12:
            data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
        elif num_leads == 6:                                     
            data = data[[0,1,2,3,4,5]]
        elif num_leads == 4:
            data = data[[0,1,2,7]]
        elif num_leads == 3:
            data = data[[0,1,7]]
        elif num_leads == 2:
            data = data[[0,1]]
        elif num_leads == 1:
            data = data[[1]]

        for i in range(data.shape[0]):
            lead_data = data[i, :]
            lead_series = pd.Series(lead_data)
            lead_series = lead_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            data[i, :] = lead_series.values
            data[i, :] = standardize_signal(data[i, :])

        data = data.reshape(data.shape[1], data.shape[0])
        filtered_X_data.append(data)

    return np.array(filtered_X_data)
        
def generate_X(X_data) :
    while True :
        for data in X_data:
            yield data





def batch_generator_2(batch_size, gen_x, gen_x2, gen_y, num_leads, num_classes, cc_data_len): 
    #np.random.shuffle(order_array)
    batch_cc = np.zeros((batch_size, cc_data_len))
    batch_features = np.zeros((batch_size,5000, num_leads))
    batch_labels = np.zeros((batch_size,num_classes))
    while True:
        for i in range(batch_size):
            batch_cc[i] = next(gen_x2)
            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
        batch_features_comb = [batch_features, batch_cc]
        yield batch_features_comb, batch_labels  

def batch_generator_1_train(batch_size, gen_x, gen_y, num_leads, num_classes): 
    batch_features = np.zeros((batch_size, 5000, num_leads))  # ECG 데이터 배치
    batch_labels = np.zeros((batch_size, num_classes))  # 라벨 배치

    while True:
        for i in range(batch_size):
            batch_features[i] = next(gen_x)  # ECG 데이터만 가져옴
            batch_labels[i] = next(gen_y)  # 라벨 가져옴

        mix_batch_features, mix_batch_labels = flow_mixup(batch_features, batch_labels)

        yield mix_batch_features, mix_batch_labels  # 변경된 배치 반환

def batch_generator_1(batch_size, gen_x, gen_y, num_leads, num_classes): 
    batch_features = np.zeros((batch_size, 5000, num_leads))  # ECG 데이터 배치
    batch_labels = np.zeros((batch_size, num_classes))  # 라벨 배치

    while True:
        for i in range(batch_size):
            batch_features[i] = next(gen_x)  # ECG 데이터만 가져옴
            batch_labels[i] = next(gen_y)  # 라벨 가져옴

        yield batch_features, batch_labels  # 변경된 배치 반환


def flow_mixup(inputs, targets, alpha=0.3, num_minority_classes=10):
    """
    Args:
        inputs: Tensor of shape (B, T, C) or (B, C, T)
        targets: Tensor of shape (B, num_classes) — one-hot or multi-hot
        alpha: Beta distribution parameter
        num_minority_classes: how many least frequent classes to focus on
    Returns:
        Mixed inputs and targets of shape (2B, ...)
    """
    # Ensure dtype consistency
    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)

    B = tf.shape(inputs)[0]
    class_counts = tf.reduce_sum(targets, axis=0)  # (num_classes,)
    _, class_order = tf.nn.top_k(-class_counts, k=tf.shape(class_counts)[0])  # 내림차순 정렬
    minority_class_indices = class_order[-num_minority_classes:]  # 뒤에서 10개

    # 해당 소수 클래스가 포함된 샘플만 선택
    minority_mask = tf.reduce_sum(tf.gather(targets, indices=minority_class_indices, axis=1), axis=1) > 0
    minority_idx = tf.where(minority_mask)

    if tf.size(minority_idx) == 0:
        # 소수 클래스가 없으면 그대로 반환
        return tf.concat([inputs, inputs], axis=0), tf.concat([targets, targets], axis=0)

    minority_idx = tf.reshape(minority_idx, [-1])
    n_m = tf.shape(minority_idx)[0]

    # 샘플 재조합
    perm = tf.random.shuffle(minority_idx)
    
    lam = tf.random.uniform([n_m], minval=0.01, maxval=0.99, dtype=tf.float32)
    lam = tf.reshape(lam, [-1, 1, 1])
    lam_y = tf.reshape(lam, [-1, 1])

    input_m = tf.gather(inputs, minority_idx)
    input_p = tf.gather(inputs, perm)
    target_m = tf.gather(targets, minority_idx)
    target_p = tf.gather(targets, perm)

    mixed_input = lam * input_m + (1 - lam) * input_p
    mixed_target = lam_y * target_m + (1 - lam_y) * target_p

    # 전체 배치 결합
    final_inputs = tf.concat([inputs, mixed_input], axis=0)
    final_targets = tf.concat([targets, mixed_target], axis=0)

    return final_inputs, final_targets







