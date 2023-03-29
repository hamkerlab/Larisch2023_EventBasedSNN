import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

def test_Jitter(t,clf):

    ## test set
    
    data_test = np.load('output_svm/drop_event_t_%i/spkC_V1_test.npy'%(t))
    labels_test =  np.load('output_svm/drop_event_t_%i/labels_test.npy'%(t))

    print(np.shape(data_test), np.shape(labels_test))
    n_samples, n_patches, n_cells = np.shape(data_test)

    data_test = np.reshape(data_test,(n_samples,n_patches*n_cells))

    p = clf.predict(data_test)

    acc_score = accuracy_score(labels_test,p)*100.
    print(acc_score)
    
    np.savetxt('accuracy_%i.txt'%(t),[acc_score])

    print(confusion_matrix(labels_test,p, normalize='true'))

    plot_confusion_matrix(clf,data_test,labels_test )
    plt.savefig('output_svm/drop_event_t_%i/conf_M'%(t))

def main():
    print('Start to fit a SVM on the spike counts of the normal training set')
    ## load training set
    data_training = np.load('output_svm/spkC_V1_train.npy')
    labels_train = np.load('output_svm/labels_train.npy')

    print(np.shape(data_training), np.shape(labels_train))
    n_samples, n_patches, n_cells = np.shape(data_training)

    data_training = np.reshape(data_training,(n_samples,n_patches*n_cells))

    n_classes = 10

    clf = LinearSVC(C=1.0,multi_class='ovr',verbose=0, max_iter=3000)

    clf.fit(data_training,labels_train)
    print('Get the prediction on the test-set with different levels of drop-Event rates')
    
    std_t = np.linspace(0,100000,11)
    for t in range(len(std_t)):
        test_Jitter(t, clf) 


if __name__ == "__main__":

    main()
