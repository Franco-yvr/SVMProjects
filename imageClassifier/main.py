from util import load, build_vocabulary, get_bags_of_sifts, showAverageHistogram, heat_confusion_matrix, accuracyScore
from classifiers import nearest_neighbor_classify, svm_classify

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = load("sift/train")
test_image_paths, test_labels = load("sift/test")
        
print('Extracting SIFT features\n')
kmeans = build_vocabulary(train_image_paths, vocab_size=200)

train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)

print('Using support vector machine to predict test set categories\n')
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)

print('---Evaluation---\n')
showAverageHistogram(train_image_feats, train_labels, train_image_paths)
# score of nearest neighbor classifier with bag of SIFT representation

# score of support vector machines predictions with bag of SIFT representation
print("The score of support vector machines predictions with bag of SIFT representation is: ", accuracyScore(test_labels, pred_labels_svm))

# plot 15x15 confusion matrix of support vector machines predictions with bag of SIFT representation
title = 'Result for support vector machine classifier with bag of SIFT'
heat_confusion_matrix(test_image_paths, test_labels, pred_labels_svm, title)
