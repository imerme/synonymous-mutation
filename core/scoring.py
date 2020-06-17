import sklearn.metrics as metrics
import numpy as np
import pandas as pd


#s en, spe, pre, f1, mcc, acc,auc,AUPR,tn,fp,fn,tp
def get_scores(y_true, y_score, threshold=0.5, return_dict=True):
    y_pred=[int(i>=threshold) for i in y_score]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.flatten()
    sen = tp/(fn+tp)
    spe = tn/(fp+tn)
    pre = metrics.precision_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_score)
    pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(rc, pr)
    f1 = metrics.f1_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    if return_dict:
        return dict(sen=sen, spe=spe, pre=pre, f1=f1, mcc=mcc, acc=acc, auc=auc, aupr=aupr, tn=tn, fp=fp, fn=fn, tp=tp)
    return np.array([sen, spe, pre, f1, mcc, acc, auc, aupr, tn, fp, fn, tp])


def show_scores(train_out_evalue, val_out_evalue, test_out_evalue, file):
    pd.set_option('display.width',1000)
    train_out_evalue_ = np.concatenate([np.array(train_out_evalue), np.array(train_out_evalue).mean(axis=0).reshape(1, 12)])
    columns = ['sen', 'spe', 'pre', 'f1', 'mcc', 'acc', 'auc', 'aupr', 'tn', 'fp', 'fn', 'tp']
    df = pd.DataFrame(data=train_out_evalue_, columns=columns)
    print(df)
    val_out_evalue_ = np.concatenate([np.array(val_out_evalue), np.array(val_out_evalue).mean(axis=0).reshape(1, 12)])
    df = pd.DataFrame(data=val_out_evalue_, columns=columns)
    df.to_csv(file)
    print(df)
    test_out_etestue_ = np.concatenate([np.array(test_out_evalue), np.array(test_out_evalue).mean(axis=0).reshape(1, 12)])
    df = pd.DataFrame(data=test_out_etestue_, columns=columns)
    print(df)


def parse_scores(train_out_scores, val_out_scores, test_out_scores):
    scores = []
    train_means = np.array(train_out_scores).mean(axis=0).reshape(1, 12)
    val_means = np.array(val_out_scores).mean(axis=0).reshape(1, 12)
    test_means = np.array(test_out_scores).mean(axis=0).reshape(1, 12)
    scores.append(train_means)
    scores.append(val_means)
    scores.append(test_means)
    return scores

def ps(scores):
    return np.array(scores).mean(axis=0).reshape(1, 12)


class scob():
    @classmethod
    def get_scores(cls, y_true, y_score, threshold=0.5):
        y_pred = [int(i >= threshold) for i in y_score]
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix.flatten()
        sen = tp / (fn + tp)
        spe = tn / (fp + tn)
        pre = metrics.precision_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_score)
        pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr = metrics.auc(rc, pr)
        f1 = metrics.f1_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        acc = metrics.accuracy_score(y_true, y_pred)
        return np.array([sen, spe, pre, f1, mcc, acc, auc, aupr, tn, fp, fn, tp])

    @classmethod
    def get_scores_from_cv_out(cls, source, file=None, label_1='output_score', label_2='train_out',
                               return_mean_only=False):
        out_scores = []
        for (y_true, y_pred) in source:
            out_scores.append(cls.get_scores(y_true, y_pred))
        out_scores = np.array(out_scores)
        cv = len(out_scores)
        columns = ['sen', 'spe', 'pre', 'f1', 'mcc', 'acc', 'auc', 'aupr', 'tn', 'fp', 'fn', 'tp']
        mean = np.array(out_scores).mean(axis=0).reshape(1, 12)
        if return_mean_only:
            return mean
        out_score_ = np.concatenate(
            [out_scores, out_scores.mean(axis=0).reshape(1, 12)])
        index_1 = [label_1]
        index_2 = [label_2 + 'cv' + str(i) for i in range(1, cv + 1)]
        index_2.append(label_2 + '_mean')
        scores = pd.DataFrame(out_score_, columns=columns,
                              index=pd.MultiIndex.from_product([index_1, index_2]))
        return scores

    @classmethod
    def concat_scores(cls, *scores,
                      columns=['sen', 'spe', 'pre', 'f1', 'mcc', 'acc', 'auc', 'aupr', 'tn', 'fp', 'fn', 'tp'],
                      index=[[], []]):
        df = pd.DataFrame(columns=columns, index=index)
        for score in scores:
            df = df.append(score)
        return df



