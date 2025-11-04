from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, hinge_loss
from sklearn.utils.class_weight import compute_sample_weight
import time

def train_decision_tree(X_train, y_train, X_test, y_test):
    w_train = compute_sample_weight('balanced', y_train)
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=35)
    t0 = time.time()
    dt_clf.fit(X_train, y_train, sample_weight=w_train)
    dt_time = time.time() - t0
    dt_pred = dt_clf.predict_proba(X_test)[:, 1]
    dt_roc_auc = roc_auc_score(y_test, dt_pred)
    return dt_clf, dt_time, dt_roc_auc

def train_svm(X_train, y_train, X_test, y_test):
    svm_clf = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False, max_iter=10000)
    t0 = time.time()
    svm_clf.fit(X_train, y_train)
    svm_time = time.time() - t0
    svm_pred = svm_clf.decision_function(X_test)
    svm_roc_auc = roc_auc_score(y_test, svm_pred)
    svm_hinge = hinge_loss(y_test, svm_pred)
    return svm_clf, svm_time, svm_roc_auc, svm_hinge
