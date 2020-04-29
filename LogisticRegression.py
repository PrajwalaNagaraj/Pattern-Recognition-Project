
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve

random.seed(0)

# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
path_to_data = "data/imdb/"
method = 0


def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos,test_neg)

        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    print("Logistic Regression")
    print("-------------------")
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)

# Loads the train and test set into four different lists.
def load_data(path_to_dir):
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir + "train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_pos.append(words)
    with open(path_to_dir + "train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_neg.append(words)
    with open(path_to_dir + "test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_pos.append(words)
    with open(path_to_dir + "test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

# Returns the feature vectors for all text in the train and test datasets.
def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):

    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    pos_total_dict = {}
    for line in train_pos:
        pos_word_dict = {}
        for word in line:
            pos_word_dict[word] = 1
        for word in pos_word_dict.keys():
            if word in pos_total_dict:
                pos_total_dict[word] = pos_total_dict[word] + 1
            else:
                pos_total_dict[word] = 1

    neg_total_dict = {}
    for line in train_neg:
        neg_word_dict = {}
        for word in line:
            neg_word_dict[word] = 1
        for word in neg_word_dict.keys():
            if word in neg_total_dict:
                neg_total_dict[word] = neg_total_dict[word] + 1
            else:
                neg_total_dict[word] = 1
    pos_without_sw = remove_stopwords(pos_total_dict, stopwords)
    neg_without_sw = remove_stopwords(neg_total_dict, stopwords)

    at_least_one_words_pos = at_least_one(pos_without_sw, train_pos)
    at_least_one_words_neg = at_least_one(neg_without_sw, train_neg)

    final_dict = at_least_twice(at_least_one_words_pos, at_least_one_words_neg, pos_total_dict, neg_total_dict)

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    feature_list = final_dict.keys()

    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for line in train_pos:
        line_dict = {}
        for word in line:
            line_dict[word] = 1
        vector_list = []
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        train_pos_vec.append(vector_list)

    for line in train_neg:
        line_dict = {}
        for word in line:
            line_dict[word] = 1
        vector_list = []
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        train_neg_vec.append(vector_list)

    for line in test_pos:
        line_dict = {}
        for word in line:
            line_dict[word] = 1
        vector_list = []
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        test_pos_vec.append(vector_list)

    for line in test_neg:
        line_dict = {}
        for word in line:
            line_dict[word] = 1
        vector_list = []
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        test_neg_vec.append(vector_list)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def remove_stopwords(total_dict, stopwords):
    filtereddict = {k: v for k, v in total_dict.items() if k not in stopwords}
    return filtereddict


def at_least_one(without_sw, train):
    threshold = len(train) * 0.01
    reqlist = {k: v for k, v in without_sw.items() if v >= threshold}
    return reqlist


def at_least_twice(words_pos, words_neg, pos_total_dict, neg_total_dict):
    final_dict = {}

    for word in words_pos.keys():
        if not word in neg_total_dict:
            final_dict[word] = 1
        elif words_pos[word] >= 2 * neg_total_dict[word]:
            final_dict[word] = 1

    # print len(final_dict)
    for word in words_neg.keys():
        if not word in pos_total_dict:
            final_dict[word] = 1
        elif words_neg[word] >= 2 * pos_total_dict[word]:
            final_dict[word] = 1
    return final_dict


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    X = train_pos_vec + train_neg_vec
    logr_model = sklearn.linear_model.LogisticRegression()
    lr_model = logr_model.fit(X, Y)


    gnb = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model = gnb.fit(X, Y)

    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    X = train_pos_vec + train_neg_vec
    logr_model = sklearn.linear_model.LogisticRegression()
    lr_model = logr_model.fit(X, Y)

    gnb = sklearn.naive_bayes.GaussianNB()
    nb_model = gnb.fit(X, Y)

    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    predict1 = model.predict(test_pos_vec).tolist()
    predict2 = model.predict(test_neg_vec).tolist()

    tp = predict1.count('pos')
    fn = predict1.count('neg')
    tn = predict2.count('neg')
    fp = predict2.count('pos')
    accuracy = float((tp + tn)) / (len(test_pos_vec) + len(test_neg_vec))

    if print_confusion:
        print("predicted:\tpos\tneg")
        print("actual:")
        print("pos\t\t%d\t%d" % (tp, fn))
        print("neg\t\t%d\t%d" % (fp, tn))
    print("accuracy: %f" % (accuracy))

    y_true = []
    y_score = []
    for i in enumerate(predict1):
        if i == 'pos':
            y_true.append(1)
        else:
            y_true.append(0)
    for i in enumerate(predict2):
        if i == 'pos':
            y_score.append(1)
        else:
            y_score.append(0)
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC: %.2f" % pr_auc)
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: %.2f" % roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

if __name__ == "__main__":
    main()
