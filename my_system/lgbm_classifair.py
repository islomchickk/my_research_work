# from lightgbm import LGBMClassifier
# from recomendation_system.dbscan import Cluster


# class PredictModel:
#     def __init__(self):
#         self.cluster = Cluster()

#     def fit_classification_models(self, train_meta):
#         self.train_meta = self.cluster.fit(train_meta)
#         self.model_0 = LGBMClassifier(verbose=-1).fit(
#             self.train_meta[self.train_meta['cluster']==0].drop(['best_model', 'cluster'], axis=1),
#             self.train_meta[self.train_meta['cluster']==0]['best_model'])
        
#         self.model_1 = LGBMClassifier(verbose=-1).fit(
#             self.train_meta[self.train_meta['cluster']==-1].drop(['best_model', 'cluster'], axis=1), 
#             self.train_meta[self.train_meta['cluster']==-1]['best_model'])

#     def predict_best_model(self, X_test_meta):
#         X_test_cluster = self.cluster.predict_cluster(X_test_meta)
#         if X_test_cluster == 0:
#             return self.model_0.predict(X_test_meta)
#         else:
#             return self.model_1.predict(X_test_meta)


a = 19/30
b = 2/30
c = 9/30
print(a, b, c)

print(a-b, a-c)