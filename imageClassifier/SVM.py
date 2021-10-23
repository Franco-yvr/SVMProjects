from sklearn import svm

# train a linear SVM for every category (i.e. one vs all)
def svm_classify(train_image_feats, train_labels, test_image_feats, cParam):
     # create support Vector Machine  with given C value
     supportVectorMachine = svm.LinearSVC(C=cParam)

     # fit support Vector Machine with train features and labels
     supportVectorMachine.fit(train_image_feats, train_labels)

     # predict labels using support Vector Machine
     predicted_labels = supportVectorMachine.predict(test_image_feats)

     return predicted_labels
