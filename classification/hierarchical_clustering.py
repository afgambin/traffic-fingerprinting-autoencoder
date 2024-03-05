
import numpy as np
import argparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
import math as mt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Numpy file with the patterns to be clustered')
    parser.add_argument('output_labels', help='Text file to write the labels')      # output files
    parser.add_argument('output_clusters', help='Text file to write the clusters')
    parser.add_argument('validInt_file', help='Matlab file with the validInt of interest')      # comms longer than 2 hours
    parser.add_argument('mInfo_file', help='Matlab file with the mInfo')
    parser.add_argument('mTime_file', help='Matlab file with the mTime')
    args = parser.parse_args()

    encoding_sentences = np.load(args.input)
    # saving to Matlab matrix (_IP or _LTE)
    #sio.savemat('output_hlayer_IP.mat', mdict={'encoded_patterns': encoding_sentences})

    print(encoding_sentences.shape)

    print('PCA')
    pca = PCA(n_components=3)
    encoding_sentences_pca = pca.fit_transform(encoding_sentences)

    #labels = AgglomerativeClustering(n_clusters=20).fit_predict(encoding_sentences_pca)
    labels = KMeans(n_clusters=25).fit_predict(encoding_sentences_pca)
    n_clusters = len(set(labels))
    print('Number of pattern clusters: %d' % n_clusters)

    # Print the labels for each pattern
    file = open(args.output_labels, "w")
    for lab in labels:
        file.write(np.array2string(lab)+"\n")
    file.close()

    # Print the pattern indices for each label
    file = open(args.output_clusters, "w")
    for lab in set(labels):
        class_member_mask = (labels == lab)
        line = class_member_mask
        file.write("Label: " + np.array2string(lab) + " positions: ")
        num = np.argwhere(line == True)
        num = np.reshape(num, [-1])
        file.write(np.array2string(num + 1))
        file.write("\n")
    file.close()

    # Plot result
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    fig1 = plt.figure(1)
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = encoding_sentences_pca[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=10, c=tuple(col))
    plt.title('Estimated number of clusters: %d' % n_clusters)

    #plt.show()

    # User classification
    print('User classification')
    winLen = 3600
    winStep = 600
    validInt_struct = sio.loadmat(args.validInt_file)
    validInts = validInt_struct['validInts']
    expInd_range = np.linspace(0, validInts.shape[1]-1, validInts.shape[1], dtype=int)

    mInfo_struct = sio.loadmat(args.mInfo_file)
    mInfo = mInfo_struct['mInfo']
    #mInfo = mInfo_struct['mInfoLte']

    mTime_struct = sio.loadmat(args.mTime_file)
    mt1 = mTime_struct['mTime']
    #mt1 = mTime_struct['mTimeLte']

    lenThres = 20
    cnt = 0
    trSetFraction = 0.8

    trSet = np.zeros((1, n_clusters+1), dtype=int)
    evSet = np.zeros((1, n_clusters+1), dtype=int)

    for expInd in expInd_range:
        validInt = validInts[0, expInd]
        for ii in np.linspace(0, len(validInt)-1, len(validInt), dtype=int):
            patternTime = mt1[np.argwhere(mInfo[:, -1] == expInd+1), 1]
            patternList = labels[np.argwhere(mInfo[:, -1] == expInd+1)]
            valid_indices = np.argwhere((patternTime >= validInt[ii, 0]) & (patternTime <= validInt[ii, 1]))
            patternList = patternList[valid_indices[:, 0]]
            patternTime = patternTime[valid_indices[:, 0]]
            patternTime = patternTime - np.amin(patternTime)
            winNum = mt.floor((np.amax(patternTime) - winLen) / winStep)
            tmpFreq = np.zeros((winNum, n_clusters))
            for jj in np.linspace(0, winNum-1, winNum, dtype=int):
                valid_indices = np.argwhere((patternTime > jj * winStep) & (patternTime <= (jj * winStep + winLen)))
                valid_patternList = patternList[valid_indices[:, 0]]
                valid_patternList = np.reshape(valid_patternList, [-1])
                histog = np.histogram(valid_patternList, bins=n_clusters)
                tmpFreq[jj, :] = histog[0]
            #patternFreq{expInd} = [patternFreq{expInd}; tmpFreq]
            recordLen = len(tmpFreq)
            if recordLen > lenThres:
                cnt = cnt + 1
                arr = cnt * np.ones((recordLen, 1), dtype=int)
                tmpData = np.zeros((recordLen, n_clusters+1), dtype=int)
                tmpData[:, 0] = np.transpose(arr)
                tmpData[:, 1:] = tmpFreq
                trLen = mt.ceil(recordLen * trSetFraction)
                evLen = recordLen - trLen

                # randmization
                trSet = np.concatenate((trSet, tmpData[0: trLen, :]))
                evSet = np.concatenate((evSet, tmpData[(trLen + 1):-1, :]))
    trSet = trSet[1:, :]  # cancel the empty row
    evSet = evSet[1:, :]  # cancel the empty row

    userTarget = np.array([1, 2, 2, 1, 3, 3, 3, 2, 4, 4])

    # SVC
    clf = SVC()
    clf.fit(trSet[:, 1:], userTarget[trSet[:, 0]])
    trAccuracy = clf.score(trSet[:, 1:], userTarget[trSet[:, 0]])
    print('Training accuracy SVC:: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], userTarget[evSet[:, 0]])
    print('Validation accuracy SVC:: %f' % evAccuracy)

    # KNN
    clf = neighbors.KNeighborsClassifier(15)
    clf.fit(trSet[:, 1:], userTarget[trSet[:, 0]])
    trAccuracy = clf.score(trSet[:, 1:], userTarget[trSet[:, 0]])
    print('Training accuracy KNN: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], userTarget[evSet[:, 0]])
    print('Validation accuracy KNN: %f' % evAccuracy)

    # Neural network
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)
    clf.fit(trSet[:, 1:], userTarget[trSet[:, 0]])
    trAccuracy = clf.score(trSet[:, 1:], userTarget[trSet[:, 0]])
    print('Training accuracy neural network: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], userTarget[evSet[:, 0]])
    print('Validation accuracy neural network: %f' % evAccuracy)