from torch.nn.functional import one_hot
from numbers import Number
import numpy as np
import torch

from .accuracy import accuracy

def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes**2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix.detach().cpu().numpy()
    
def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    num_classes = pred.size(1) # size 0为图片数量，size 1 为类别数量
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0).detach().cpu().numpy()
        precision = class_correct / np.maximum(pred_positive.sum(0).detach().cpu().numpy(), 1.) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0).detach().cpu().numpy(), 1.) * 100
        f1_score = 2 * precision * recall / np.maximum(
            precision + recall,
            torch.finfo(torch.float32).eps)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == 'none':
            precision = precision
            recall = recall
            f1_score = f1_score
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0].tolist(), recalls[0].tolist(), f1_scores[0].tolist()
    else:
        return precisions, recalls, f1_scores

def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().detach().cpu().numpy())
        elif average_mode == 'none':
            res = res.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res
    
def evaluate(
                results,
                gt_labels,
                metric='accuracy',
                metric_options=None,
                indices=None,
                logger=None):
    """Evaluate the dataset.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `accuracy`.
        metric_options (dict, optional): Options for calculating metrics.
            Allowed keys are 'topk', 'thrs' and 'average_mode'.
            Defaults to None.
        indices (list, optional): The indices of samples corresponding to
            the results. Defaults to None.
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Defaults to None.
    Returns:
        dict: evaluation results
    """
    if metric_options is None:
        metric_options = {'topk': (1, 5)}
    if isinstance(metric, str):
        metrics = [metric]
    else:
        metrics = metric
    allowed_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'support','confusion'
    ]
    eval_results = {}
    #results = np.vstack(results)
    # gt_labels = self.get_gt_labels()
    if indices is not None:
        gt_labels = gt_labels[indices]
    num_imgs = len(results)
    assert len(gt_labels) == num_imgs, 'dataset testing results should '\
        'be of the same length as gt_labels.'

    invalid_metrics = set(metrics) - set(allowed_metrics) # 判断metrics是否存在
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    topk = metric_options.get('topk', (1, 5)) # 若不存在则设置为(1,5)
    
    # 判断数据集类别是否小于5，若小于5则top5为100%，但为避免索引报错，将topk最大值取类别数
    if max(topk) > len(results[0]):
        topk = (1,)
    
    thrs = metric_options.get('thrs')         # 不存在为None
    average_mode = metric_options.get('average_mode', 'macro')

    if 'accuracy' in metrics:
        if thrs is not None:
            acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
        else:
            acc = accuracy(results, gt_labels, topk=topk)
        if isinstance(topk, tuple):
            eval_results_ = {
                f'accuracy_top-{k}': a
                for k, a in zip(topk, acc)
            }
        else:
            eval_results_ = {'accuracy': acc}
        # 返回的acc为嵌套列表[[]]，“子”列表长度与thrs对应。比如acc=[[98]],thrs长度则为1，在本次程序中设置至少为(0.)。
        # 若acc=[[98,100]]，即thrs为长度为2，比如(0.1,0.2)。acc“总”长度与topk对应，此处是让结果相互对应。
        # 比如此处的结果可以为{
        #     accuracy_top-1—_thr_0.15=92.37,
        #     accuracy_top-1—_thr_0.2=90.37,
        #     accuracy_top-5—_thr_0.15=95.37,
        #     accuracy_top-5—_thr_0.2=94.37,
        # }
        if isinstance(thrs, tuple): 
            for key, values in eval_results_.items():
                eval_results.update({
                    f'{key}_thr_{thr:.2f}': value.item()
                    for thr, value in zip(thrs, values)
                })
        else:
            # 如果为空，则把前述获得的acc转为数值，因为取得的为Tensor
            eval_results.update(
                {k: v.item()
                    for k, v in eval_results_.items()})

    if 'support' in metrics:
        support_value = support(
            results, gt_labels, average_mode=average_mode)
        eval_results['support'] = support_value

    if 'confusion' in metrics:
        confusion_matrix = calculate_confusion_matrix(
            results, gt_labels)
        eval_results['confusion'] = confusion_matrix

    precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
    if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
        if thrs is not None:
            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode, thrs=thrs)
        else:
            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode)
        for key, values in zip(precision_recall_f1_keys,
                                precision_recall_f1_values):
            if key in metrics:
                if isinstance(thrs, tuple):
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value
                        for thr, value in zip(thrs, values)
                    })
                else:
                    eval_results[key] = values

    return eval_results
