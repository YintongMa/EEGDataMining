import numpy as np
import pandas as pd
from scipy.fft import fft
import eeglib as eeg
import warnings
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation


def test(*args):
    for a in args:
        print(a)
    exit()


channel_list = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


def extract_features(data):
    matrix = np.array([data[c][0] for c in channel_list])

    warnings.filterwarnings(action="ignore", category=np.ComplexWarning)
    matrix = np.array([fft(row) for row in matrix]).astype(float)

    freq = [i / 128 for i in range(1, 257)]
    features = np.zeros((len(channel_list), 25))

    for i in range(len(matrix)):
        row = matrix[i]
        cur = features[i]

        cur[0] = eeg.features.hjorthActivity(row)
        cur[1] = eeg.features.hjorthComplexity(row)
        cur[2] = eeg.features.hjorthMobility(row)
        cur[3] = eeg.features.sampEn(row)
        cur[4] = eeg.features.PFD(row)
        cur[5] = eeg.features.HFD(row)
        cur[6] = eeg.features.countSignChanges(row)
        cur[7] = eeg.features.LZC(row)
        cur[8] = eeg.features.DFA(row)

        rqa = get_rqa(row)
        cur[9] = rqa.recurrence_rate
        cur[10] = rqa.determinism
        cur[11] = rqa.average_diagonal_line
        cur[12] = rqa.longest_diagonal_line
        cur[13] = rqa.divergence
        cur[14] = rqa.entropy_diagonal_lines
        cur[15] = rqa.laminarity
        cur[16] = rqa.trapping_time
        cur[17] = rqa.longest_vertical_line
        cur[18] = rqa.entropy_vertical_lines
        cur[19] = rqa.average_white_vertical_line
        cur[20] = rqa.longest_white_vertical_line
        cur[21] = rqa.longest_white_vertical_line_inverse
        cur[22] = rqa.entropy_white_vertical_lines
        cur[23] = rqa.ratio_determinism_recurrence_rate
        cur[24] = rqa.ratio_laminarity_determinism



    return features



def get_rqa(data):
    time_series = TimeSeries(data,
                             embedding_dimension=2,
                             time_delay=1)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(0.65),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    computation = RQAComputation.create(settings,
                                        verbose=True)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2

    return result





data = pd.read_csv('EP1.01.txt', sep='\t',
                   names=['id', 'event', 'device', 'channel', 'code', 'size', 'data'],
                   dtype={'id':int, 'event':int,
                          'device':str, 'channel':str,
                          'code':int, 'size':int,
                          'data':object})

data = data[data['size'] == 256]
data = data.drop(['device','id','size'], axis=1)
data['data'] = data['data'].apply(lambda x: [float(i) for i in x.split(',')])



epoch_list = []
event_list = data.event.unique()

for e in event_list:
    cur = data[data.event==e]
    info = cur[['channel', 'data']]
    epoch = {'id':e, 'code':cur.code.iat[0],
           'data':info.set_index('channel').T.to_dict('list')}
    if len(epoch['data'].keys())==14:
        epoch_list.append(epoch)



x = np.zeros((len(epoch_list), 14, 25))
y = np.zeros(len(epoch_list))

for i in range(len(epoch_list)):
    e = epoch_list[i]
    y = np.append(y, e['code'])
    features = extract_features(e['data'])
    x[i] = features



np.savez('eeg_data', x=x, y=y)











