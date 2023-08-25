from typing import Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric
from sklearn.metrics import roc_curve

from mmpretrain.registry import METRICS


@METRICS.register_module()
class ACER(BaseMetric):
    default_prefix: Optional[str] = 'acer'

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data_sample in data_samples:
            self.results.append({
                'pred_score': data_sample['pred_score'],
                'gt_label': data_sample['gt_label'],
            })

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        gt_label = torch.cat([result['gt_label'] for result in results])
        pred_score = torch.stack([result['pred_score'] for result in results])

        best_acer, best_apcer, best_bpcer, best_threshold, eer_threshold, eer = self.calculate(pred_score, gt_label)
        return {
            'best_acer': best_acer.item(),
            'best_apcer': best_apcer.item(),
            'best_bpcer': best_bpcer.item(),
            'best_threshold': best_threshold.item(),
            'eer_threshold': eer_threshold.item(),
            'eer': eer.item(),
        }

    @staticmethod
    def calculate(
            pred_score: torch.Tensor,
            gt_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert pred_score.ndim == 2 and pred_score.shape[1] == 2

        fpr, tpr, thresholds = roc_curve(gt_label.cpu().numpy(), pred_score[:, 1].cpu().numpy(), pos_label=1)
        fpr, tpr, thresholds = torch.from_numpy(fpr), torch.from_numpy(tpr), torch.from_numpy(thresholds)
        fnr = 1 - tpr
        tnr = 1 - fpr
        num_fake, num_live = torch.bincount(gt_label)

        tp = tpr * num_live
        fn = num_live - tp
        tn = tnr * num_fake
        fp = num_fake - tn

        apcer = fp / num_fake
        bpcer = fn / num_live
        acer = (apcer + bpcer) / 2

        best_acer = torch.min(acer)
        best_acer_index = torch.argmin(acer)
        best_apcer = apcer[best_acer_index]
        best_bpcer = bpcer[best_acer_index]
        best_threshold = thresholds[best_acer_index]

        eer_index = torch.argmin(torch.abs(fnr - fpr))
        eer_threshold = thresholds[eer_index]
        eer_from_fpr = fpr[eer_index]
        eer_from_fnr = fnr[eer_index]
        eer = (eer_from_fpr + eer_from_fnr) / 2

        return best_acer, best_apcer, best_bpcer, best_threshold, eer_threshold, eer
