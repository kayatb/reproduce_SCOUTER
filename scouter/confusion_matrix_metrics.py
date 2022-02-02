"""
Calculate the following confusion matrix metrics:
Area Under Curve, accuracy, precision, recall, F1-score and Kappa for the given model.

In the experiments, these metrics are only reported (and thus implemented) for the ACRIMA dataset.
"""

import argparse
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score


from train import get_args_parser
from dataset.ACRIMA import get_data, ACRIMA
from dataset.transform_func import make_transform
from sloter.slot_model import SlotModel


def calc_metrics(model, imgs, labels):
    """Calculate all metrics."""
    model.eval()

    # Obtain the model predictions.
    with torch.no_grad():
        pred_probs = model(imgs)

    preds = torch.argmax(pred_probs, dim=1)

    # Calculate the confusion matrix.
    tn, fp, fn, tp = confusion_matrix(labels.cpu(), preds.cpu()).ravel()

    # Calculate the metrics.
    metrics = {}
    metrics["auc"] = roc_auc_score(labels.cpu(), pred_probs[:, 1].cpu())
    metrics["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    metrics["recall"] = tp / (tp + fn)
    metrics["precision"] = tp / (tp + fp)
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    metrics["kappa"] = cohen_kappa_score(labels.cpu(), preds.cpu())

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("model training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    assert args.dataset == "ACRIMA", "Calculating the metrics is only implemented for the ACRIMA dataset."

    args_dict = vars(args)
    args_for_evaluation = ["num_classes", "lambda_value", "power", "slots_per_class"]
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    device = torch.device(args.device)

    # Retrieve the data.
    _, val_data = get_data(args.dataset_dir)
    val_dataset = ACRIMA(val_data, transform=make_transform(args, "val"))
    # Use the whole length as batch size, to process all images at the same time.
    # The ACRIMA dataset is small, so this should be no problem.
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, len(val_dataset), shuffle=False, num_workers=1, pin_memory=True
    )

    # Load the model from checkpoint.
    model_name = (
        f"{args.dataset}_"
        + f"{'use_slot_' if args.use_slot else 'no_slot_'}"
        + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"
        + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}"
        + "checkpoint.pth"
    )
    model = SlotModel(args)
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # Calculate the metrics.
    for batch in data_loader_val:  # Should be only a single batch.
        imgs = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.int8)
        metrics = calc_metrics(model, imgs, labels)
        print(metrics)
