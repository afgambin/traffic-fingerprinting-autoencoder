
import numpy as np
import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.mixture import BayesianGaussianMixture
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
import math as mt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file with the patterns to be clustered')
    parser.add_argument('output_labels', help='Text file to write the labels')
    parser.add_argument('output_clusters', help='Text file to write the clusters')
    parser.add_argument('validInt_file', help='Matlab file with the validInt of interest')
    parser.add_argument('mInfo_file', help='Matlab file with the mInfo')
    parser.add_argument('mTime_file', help='Matlab file with the mTime')
    args = parser.parse_args()

    encoding_sentences_struct = sio.loadmat(args.input)
    encoding_sentences = encoding_sentences_struct['mPack']
    appSet = np.argwhere(encoding_sentences[:, 0] <= 0)
    serSet = np.argwhere(encoding_sentences[:, 0] > 0)
    encoding_sentences = encoding_sentences[:, 2:]
    encoding_sentences = encoding_sentences.astype(int)
    # for each in encoding_sentences:
    #     for ii in each:
    #         ii = np.binary_repr(ii, 4)
    print(encoding_sentences.shape)

    print('\n---Reducing the dimensionality---')
    n_PCA = 2
    pca = PCA(n_components=n_PCA)
    encoding_sentences_pca_app = pca.fit_transform(np.log10(encoding_sentences[appSet[:, 0], :] + 1))
    encoding_sentences_pca_ser = pca.fit_transform(np.log10(encoding_sentences[serSet[:, 0], :] + 1))
    encoding_sentences_pca = np.zeros([encoding_sentences.shape[0], n_PCA])
    encoding_sentences_pca[appSet[:, 0], :] = encoding_sentences_pca_app
    encoding_sentences_pca[serSet[:, 0], :] = encoding_sentences_pca_ser

    print('\n---Clustering---')
    app_clusters = 12
    ser_clusters = 6
    n_clusters = app_clusters + ser_clusters
    labels_app = KMeans(n_clusters=app_clusters, algorithm='elkan', n_init=5).fit_predict(encoding_sentences_pca_app)
    labels_ser = KMeans(n_clusters=ser_clusters, algorithm='elkan', n_init=5).fit_predict(encoding_sentences_pca_ser)
    labels = np.zeros(encoding_sentences.shape[0])
    labels[appSet[:, 0]] = labels_app
    labels[serSet[:, 0]] = labels_ser + app_clusters
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
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(set(labels)))]
    fig1 = plt.figure(1)
    plt.plot([3, 1, 3])
    plt.subplot(131)
    for k, col in zip(set(labels_app), colors):
        class_member_mask = (labels == k)
        xy = encoding_sentences_pca[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=10, c=tuple(col))
        plt.title('Application')
    plt.subplot(132)
    for k, col in zip(set(labels_ser + 12), colors[11:]):
        class_member_mask = (labels == k)
        xy = encoding_sentences_pca[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=10, c=tuple(col))
        plt.title('Server')
    plt.subplot(133)
    for k, col in zip(set(labels), colors):
        class_member_mask = (labels == k)
        xy = encoding_sentences_pca[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=10, c=tuple(col))
        plt.title('Server and application')

    plt.show()

    # Classification
    print('\n---Start classification---')
    winLen = 3600
    winStep = 600
    validInt_struct = sio.loadmat(args.validInt_file)
    validInts = validInt_struct['validInts']
    expInd_range = np.linspace(0, validInts.shape[1]-1, validInts.shape[1], dtype=int)

    mInfo_struct = sio.loadmat(args.mInfo_file)
    mInfo = mInfo_struct['mInfo']
    mInfo = mInfo[0:len(labels)]

    mTime_struct = sio.loadmat(args.mTime_file)
    mt1 = mTime_struct['mTime']
    mt1 = mt1[0:len(labels)]

    lenThres = 20
    cnt = 0
    trSetFraction = 0.8

    trSet = np.zeros((1, n_clusters+1), dtype=int)
    evSet = np.zeros((1, n_clusters+1), dtype=int)

    for expInd in expInd_range:
        validInt = validInts[0, expInd]
        patternFreq = np.zeros((1, n_clusters), dtype='float64')
        for ii in np.linspace(0, validInt.shape[0] - 1, validInt.shape[0], dtype=int):
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
            patternFreq = np.concatenate((patternFreq, tmpFreq))
        patternFreq = patternFreq[1:, :]  # cancel the empty row

        # remove the starting slot if it is larger than this factor of the average value
        startFactor = 1.5
        startDuration = 20  # minutes
        ii_range = np.linspace(0, mt.ceil(startDuration * 60 / winStep)-1, mt.ceil(startDuration * 60 / winStep), dtype=int)
        ii_range = np.flip(ii_range, axis=0)
        for ii in ii_range:
            patternFreq_temp = patternFreq[(ii + 1):, :]
            two_vector = 2*np.ones((patternFreq_temp.shape[0], patternFreq_temp.shape[1]))
            sum_factor = np.sum([patternFreq_temp, two_vector], axis=0)
            condition = startFactor * np.mean(sum_factor)
            if np.sum(patternFreq[ii, :]) > condition:
                patternFreq = np.delete(patternFreq, ii, 0)

        recordLen = len(patternFreq)
        if recordLen > lenThres:
            cnt = cnt + 1
            arr = cnt * np.ones((recordLen, 1), dtype=int)
            tmpData = np.zeros((recordLen, n_clusters+1), dtype=int)
            tmpData[:, 0] = np.transpose(arr)
            tmpData[:, 1:] = patternFreq
            trLen = mt.ceil(recordLen * trSetFraction)
            evLen = recordLen - trLen

            # randomization
            trSet = np.concatenate((trSet, tmpData[0:trLen, :]))
            evSet = np.concatenate((evSet, tmpData[trLen:, :]))
    trSet = trSet[1:, :]  # cancel the empty row
    evSet = evSet[1:, :]  # cancel the empty row

    userTarget = np.array([1, 2, 2, 1, 3, 3, 3, 2, 4, 4])

    # User Classification
    print('\n---User classification---')
    
    # SVC
    clf = SVC()
    clf.fit(trSet[:, 1:], userTarget[trSet[:, 0] - 1])
    trAccuracy = clf.score(trSet[:, 1:], userTarget[trSet[:, 0] - 1])
    print('Training accuracy SVC: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], userTarget[evSet[:, 0] - 1])
    print('Validation accuracy SVC: %f' % evAccuracy)

    # KNN
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
    clf.fit(trSet[:, 1:], userTarget[trSet[:, 0] - 1])
    trAccuracy = clf.score(trSet[:, 1:], userTarget[trSet[:, 0] - 1])
    print('Training accuracy KNN: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], userTarget[evSet[:, 0] - 1])
    print('Validation accuracy KNN: %f' % evAccuracy)

    # MLP
    clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(100,), random_state=4, max_iter=100000)
    clf.fit(trSet[:, 1:], userTarget[trSet[:, 0] - 1])
    trAccuracy = clf.score(trSet[:, 1:], userTarget[trSet[:, 0] - 1])
    print('Training accuracy neural network: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], userTarget[evSet[:, 0] - 1])
    print('Validation accuracy neural network: %f' % evAccuracy)

    # Trace Classification
    print('\n---Trace classification---')

    # SVC
    clf = SVC()
    clf.fit(trSet[:, 1:], trSet[:, 0])
    trAccuracy = clf.score(trSet[:, 1:], trSet[:, 0])
    print('Training accuracy SVC: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], evSet[:, 0])
    print('Validation accuracy SVC: %f' % evAccuracy)

    # KNN
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
    clf.fit(trSet[:, 1:], trSet[:, 0])
    trAccuracy = clf.score(trSet[:, 1:], trSet[:, 0])
    print('Training accuracy KNN: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], evSet[:, 0])
    print('Validation accuracy KNN: %f' % evAccuracy)

    # MLP
    clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(100,), random_state=4, max_iter=100000)
    clf.fit(trSet[:, 1:], trSet[:, 0])
    trAccuracy = clf.score(trSet[:, 1:], trSet[:, 0])
    print('Training accuracy neural network: %f' % trAccuracy)
    predict = clf.predict(evSet[:, 1:])
    evAccuracy = clf.score(evSet[:, 1:], evSet[:, 0])
    print('Validation accuracy neural network: %f' % evAccuracy)
