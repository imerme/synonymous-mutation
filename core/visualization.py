import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp
import os


def plot_roc_curve(source, file_name):
    '''
    Plot auc curve from cross validation
    :param source: list of tuple of (y_pred, y_ture), source data for plot figures
    :param file_name: str, file to save figures
    :return: None
    '''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for i, (y_true, y_pred) in enumerate(source):
        # y_pred = y_pred[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.3f)' % (i, roc_auc))
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # plot diagonal
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Blind Guess', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_tpr[0] = 0.
    mean_auc = sum(aucs) / len(aucs)  # calculate mean_auc
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig(file_name)


def plot_roc_curve_on_ax(ax, source, label_formatter='%.3f, %.3f'):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (y_true, y_pred) in enumerate(source):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    # plot diagonal
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_tpr[0] = 0.
    mean_auc = sum(aucs) / len(aucs)  # calculate mean_auc
    std_auc = np.std(aucs)
    line, = ax.plot(mean_fpr, mean_tpr, label=label_formatter % (mean_auc, std_auc), lw=2, alpha=.8)
    return line, mean_auc, std_auc

def plot_loss_curve(history, target_file="./0308_1_loss.png"):
    '''
    Plot loss and accuracy curves, whos data from model trained with train_set and  validation_set
    params historys: dict, like {loss: list, accuracy: list, val_loss: list, val_accuracy: list}
    params target_file: str, file path to save figure
    '''
    try:
        loss, accuracy, val_loss, val_accuracy = history['loss'], history['accuracy'], history['val_loss'], history[
            'val_accuracy']
    except:
        loss, accuracy, val_loss, val_accuracy = history['loss'], history['acc'], history['val_loss'], history[
            'val_acc']
    plt.style.use("seaborn-colorblind")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(loss, color='red', lw=1, label='train_loss')
    ax.plot(accuracy, color='blue', lw=1, label='train_acc')
    ax.plot(val_loss, color='cyan', lw=1, label='val_loss')
    ax.plot(val_accuracy, color='orange', lw=1, label='val_acc')
    ax.set_title("Loss and Accuracy")
    ax.set_xlabel('Epoch #')
    ax.set_ylabel('Loss/Accuracy')
    plt.legend(loc='upper right')
    ax.set_ylim(0, 1)
    # file name
    if not target_file.endswith('.png'):
        target_file += '.png'
    plt.savefig(target_file, dpi=300)
    plt.close('all')


def plot_pr_curve(source, file_name):
    '''
    Plot auc curve from cross validation
    :param source: list of tuple of (y_pred, y_ture), source data for plot figures
    :param file_name: str, file to save figures
    :return: None
    '''

    precisions = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for i, (y_true, y_pred) in enumerate(source):
        # y_pred = y_pred[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        pr_auc = auc(recall[::-1], precision[::-1])
        precision[0] = 0
        precision[-1] = 1
        ax.plot(recall[::-1], precision[::-1], lw=1, alpha=0.3, label='PR fold %d(area=%0.3f)' % (i, pr_auc))
        precisions.append(interp(mean_recall, recall[::-1], precision[::-1]))
        pr_aucs.append(pr_auc)

    # plot diagonal
    ax.plot([0, 1], [1, 0], linestyle='--', lw=2, color='r', label='Blind Guess', alpha=.8)
    mean_precision = np.mean(precisions, axis=0)
    mean_pr_auc = sum(pr_aucs) / len(pr_aucs)  # calculate mean_auc
    std_auc = np.std(precisions, axis=0)
    ax.plot(mean_recall, mean_precision, color='b', label=r'Mean PR (area=%0.3f)' % mean_pr_auc, lw=2, alpha=.8)
    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper)
    plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.title('PR')
    plt.legend(loc='lower left')
    plt.savefig(file_name)
    plt.clf()


def plot_a_roc_curve(y_true, y_pred, file_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.plot(fpr, tpr, color='b', label=r'ROC (area=%0.3f)' % roc_auc, lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig(file_name)
    plt.close(0)


def plot_a_pr_curve(y_true, y_pred, file_name):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall[::-1], precision[::-1])
    precision[0] = 0
    precision[-1] = 1
    ax.plot(recall[::-1], precision[::-1], color='b', lw=2, alpha=.8, label='PR (area=%0.3f)' % pr_auc)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.title('PR')
    plt.legend(loc='lower left')
    plt.savefig(file_name)
    plt.close('all')


def plot_cv_models(ax, model_prototype, x, y, file_formatter, cv,
                   label_formatter=r'Mean ROC (area=%0.3f $\pm$ %0.3f)'):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for cv_idx in range(cv):
        model_prototype.load_weights(file_formatter.format(cv_idx))
        pred = model_prototype.predict(x)[:, 1]
        ture = y[:, 1]
        fpr, tpr, _ = roc_curve(ture, pred)
        roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # plot diagonal
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_tpr[0] = 0.
    mean_auc = sum(aucs) / len(aucs)  # calculate mean_auc
    std_auc = np.std(aucs)
    # test
    line, = ax.plot(mean_fpr, mean_tpr,
            label=label_formatter % (mean_auc, std_auc),
            lw=1, alpha=.8)
    return line, mean_auc, std_auc

def plot_cv_loss_curve(historys, path_to_save):
    keys = historys[1].keys()
    values = np.array([list(history.values()) for history in historys]).mean(axis=0)
    history_average = dict(zip(keys, values))
    plot_loss_curve(history_average, path_to_save)


def plot_cv_out(train_out, val_out, test_out, historys, path_to_save):
    plot_roc_curve(train_out, os.path.join(path_to_save, 'train_roc.png'))
    plot_pr_curve(train_out, os.path.join(path_to_save, 'train_pr.png'))
    plot_roc_curve(val_out, os.path.join(path_to_save, 'val_roc.png'))
    plot_pr_curve(val_out, os.path.join(path_to_save, 'val_pr.png'))
    plot_roc_curve(test_out, os.path.join(path_to_save, 'test_roc.png'))
    plot_pr_curve(test_out, os.path.join(path_to_save, 'test_pr.png'))
    #plot_cv_loss_curve(historys, os.path.join(path_to_save, 'loss_curve.png'))






