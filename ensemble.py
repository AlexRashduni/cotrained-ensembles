import numpy as np
import tensorflow as tf

class Ensemble:

    def __default_eval_agg(lst):
        lst = np.array(lst).flatten()
        uniques, inverse = np.unique(lst, return_inverse=True)
        idx = np.argmax(np.bincount(inverse))
        return uniques[idx]

    def __default_pseudolabel_func(lst):
        pl = max(set([l[0] for l in lst]), key=lst.count)
        return pl if (sum([1 if pl == lst[i] else 0 for i in range(len(lst))]) / len(lst)) >= 0.5 else None

    def __init__(self, models, eval_aggregation = __default_eval_agg):
        self.models = models
        self.eval_aggregation = eval_aggregation

    def predict(self, X, eval_aggregation = __default_eval_agg, **kwargs):
        preds = [model.predict(X, **kwargs) for model in self.models]
        return [eval_aggregation([preds[i][j] for i in range(len(preds))]) for j in range(len(preds[0]))]       

    def evaluate(self, X, Y, metric = tf.keras.metrics.MAE, eval_aggregation = __default_eval_agg):
        preds = self.predict(X, eval_aggregation=eval_aggregation, verbose=0)
        return metric(Y.flatten(), preds)
    
    def train(self, X, Y, **kwargs):
        for model in self.models:
            model.fit(X, Y, **kwargs)

    def cotrain(self, X_train, Y_train, X_test, pseudolabel_func = __default_pseudolabel_func, convergence_ratio = 0.01, max_iter = 100, **kwargs):
        iter_num = 1
        converged = False
        while (not converged) and (iter_num <= max_iter):
            self.train(X_train, Y_train, **kwargs)

            # Generates pseudolabels
            y_hat = [model.predict(X_test) for model in self.models]
            pseudolabels = [pseudolabel_func([y_hat[i][j] for i in range(len(self.models))]) for j in range(len(X_test))]
            converged = (sum([1 if not (pseudolabels[i] is None) else 0 for i in range(len(pseudolabels))]) < (convergence_ratio * len(X_test)))

            # Moves data from X_test to X_train
            np.append(X_train, [X_test[i] for i in range(len(X_test)) if not (pseudolabels[i] is None)])
            np.append(Y_train, [pseudolabels[i] for i in range(len(pseudolabels)) if not (pseudolabels[i] is None)])
            X_test = [X_test[i] for i in range(len(X_test)) if pseudolabels[i] is None]
            iter_num += 1
            converged = converged or len(X_test) == 0

        if iter_num <= max_iter:
            self.train(X_train, Y_train, epochs=max_iter-iter_num)