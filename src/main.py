import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import scipy as sp

def generateSpectrogram(fileName):
    data, fs = sf.read(fileName)

    f, t, Sxx = signal.spectrogram(data, fs)
    Sxx_log = 10 * np.log10(Sxx + 1e-20)

    plt.figure(figsize = (9, 3))
    plt.pcolormesh(t, f, Sxx_log)

    plt.gca().set_title(fileName.replace('../sentences/', ''))
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')

    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig('spectrogram.pdf')

def getFeatures(fileName):
    data, fs = sf.read(fileName)

    f, t, Sxx = signal.spectrogram(data, fs, ('tukey', 0.25), 400, 240, 511)
    Sxx_log = 10 * np.log10(Sxx + 1e-20)

    Sxxf = np.zeros((16, len(Sxx_log[0])))

    for i in range(len(Sxx[0])):
        for j in range(16):
            for k in range(j * 16, (j * 16) + 16):
                Sxxf[j][i] += Sxx_log[k][i]

    return Sxxf, t

def getRating(sentence, query):
    # Get features for both sentence and query
    # We need to transpose the matrices to match their length
    fSentence = getFeatures(sentence)[0].transpose()
    fQuery = getFeatures(query)[0].transpose()

    rating = 0
    ratingArray = []

    # Iterate the query features around sentence features and get
    # Pearson coefficients for each position of each query position
    for i in range(len(fSentence) - len(fQuery)):
        for j in range(len(fQuery)):
            rating += sp.stats.pearsonr(fQuery[j], fSentence[j + i])[0]
        
        ratingArray.append(rating)
        rating = 0
        
    # Divide each member of ratingArray by query length
    ratingArray[:] = [x / len(fQuery) for x in ratingArray]

    return ratingArray

def generateGraphs():
    sentences = ['sa1.wav', 'sa2.wav', 'si1055.wav', 'si1685.wav', 'si2315.wav', 'sx65.wav', 'sx155.wav', 'sx245.wav', 'sx335.wav', 'sx425.wav']

    for s in sentences:
        q1Rating = getRating('../sentences/' + s, '../queries/q1.wav')
        q2Rating = getRating('../sentences/' + s, '../queries/q2.wav')
        features, tf = getFeatures('../sentences/' + s)

        q1features, tq1 = getFeatures('../queries/q1.wav')
        q2features, tq2 = getFeatures('../queries/q2.wav')

        tq1_max = features.shape[1] - q1features.shape[1]
        tq2_max = features.shape[1] - q2features.shape[1]

        data, fs = sf.read('../sentences/' + s)
        t = np.arange(data.size) / fs
        
        plt.figure(figsize=(8,4))

        plt.subplot(311)
        plt.plot(t, data)
        plt.xlim(left=0, right=t[-1])
        plt.gca().set_xlabel('t')
        plt.gca().set_ylabel('signal')
        plt.gca().set_title('"Responsibility" and "Intelligence" vs. ' + s.replace('.wav', ''))

        plt.subplot(312)
        plt.pcolormesh(tf, range(0, 16), features)
        plt.gca().invert_yaxis()
        plt.xlim(left=0, right=t[-1])
        plt.gca().set_xlabel('t')
        plt.gca().set_ylabel('features')

        plt.subplot(313)
        plt.plot(tf[0:tq1_max], q1Rating, label='responsibility')
        plt.plot(tf[0:tq2_max], q2Rating, label='intelligence')
        plt.legend(loc='upper right', prop={'size': 6})
        plt.xlim(left=0, right=t[-1])
        plt.gca().set_xlabel('t')
        plt.gca().set_ylabel('scores')

        plt.tight_layout()
        plt.savefig(s.replace('.wav', '.pdf'))

        q1data, q1fs = sf.read('../queries/q1.wav')
        q2data, q2fs = sf.read('../queries/q1.wav')

        generateHits(tf[0:tq1_max], q1Rating, 0.7, s, fs, data, 'q1.wav', len(q1data))
        generateHits(tf[0:tq2_max], q2Rating, 0.75, s, fs, data, 'q2.wav', len(q2data))
        
def generateHits(time, score, threshold, s, fs, data, q, qlen):
    hits = [time[i] for i, n in enumerate(score) if n >= threshold]

    sName = s.replace('.wav', '')
    qName = q.replace('.wav', '')

    hitEnd = 0

    for h in hits:
        if h < hitEnd:
            continue
    
        hitIndex = int(h * fs)
        hitData = data[hitIndex:hitIndex + qlen]

        sf.write('../hits/' + sName + '_' + qName + '.wav', hitData, fs)
        #print('../hits/' + sName + '_' + qName + '.wav', hitIndex, hitIndex + qlen) PRINTS WHERE THE QUERY BEGINS IN A SENTENCE AND WHEN IT ENDS

        hitEnd = h + (qlen / 2)

generateGraphs()
