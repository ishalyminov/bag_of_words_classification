from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC as SVM

class ClassifierWrapper(object):
    def __init__(self):
        self.vectorizer = DictVectorizer()
        self.classifier = SVM()
        self.tfidf_transformer = TfidfTransformer()

    def train(self, in_bags, in_answers):
        bags_vectorized = self.vectorizer.fit_transform(in_bags)
        # tfidf_matrices = self.tfidf_transformer.fit_transform(bags_vectorized)
        '''tfidf_matrices'''
        self.classifier.fit(bags_vectorized, in_answers)

    def predict(self, in_bags):
        bags_vectorized = self.vectorizer.transform(in_bags)
        # tfidf_matrices = self.tfidf_transformer.transform(bags_vectorized)
        '''tfidf_matrices'''
        return self.classifier.predict(bags_vectorized)