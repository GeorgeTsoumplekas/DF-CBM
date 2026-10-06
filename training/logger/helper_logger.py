import numpy as np


def log_message(msg, logger=None, level="info"):
    """
    Unified logging wrapper. Handles standard strings or structural lists.
    If a logger is provided, it uses the specified logging level.
    Otherwise, defaults to standard print statements.
    """
    if logger is None:
        print(msg)
    else:
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(msg)


def _format_concept_table(concept_report, title):
    table_width = 84
    lines = [
        f"\n{title}",
        "=" * table_width,
        f"{'Concept':<20} | {'Acc':<8} | {'Bal-Acc':<8} | {'Prec':<7} | {'Recall':<7} | {'Macro-F1':<8} | {'AUC':<7}",
        "-" * table_width,
    ]

    for c_key, m in concept_report.items():
        if c_key == "mean":
            continue
        auc_str = f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "NaN"
        lines.append(
            f"{c_key:<20} | {m['acc'] * 100:.1f}%   | {m['bal_acc'] * 100:.1f}%   | "
            f"{m['precision']:.3f} | {m['recall']:.3f}  | {m['macro_f1']:.3f}    | {auc_str}"
        )

    lines.append("-" * table_width)
    mn = concept_report["mean"]
    mean_auc_str = f"{mn['auc']:.4f}" if not np.isnan(mn["auc"]) else "NaN"
    lines.append(
        f"{'MEAN':<20} | {mn['acc'] * 100:.1f}%   | {mn['bal_acc'] * 100:.1f}%   | "
        f"{mn['precision']:.3f} | {mn['recall']:.3f}  | {mn['macro_f1']:.3f}    | {mean_auc_str}"
    )
    lines.append("=" * table_width)
    return lines


def log_evaluation_report(
    class_m,
    concept_report,
    epoch,
    train_loss,
    logger=None,
    *,
    split_name="Test",
    video_class_m=None,
    video_concept_report=None,
):
    """
    Formats and routes the comprehensive classification summary and the
    per-concept tabular report to either a system logger or standard output.
    """
    summary_lines = [
        f"\n--> Epoch {epoch} Summary ({split_name}):",
        f"    Train Loss              : {train_loss:.4f}",
    ]

    if class_m is not None:
        summary_lines.extend(
            [
                f"    {split_name} Frame Class Acc : {class_m['acc'] * 100:.2f}%",
                f"    {split_name} Frame Class B-A : {class_m['bal_acc'] * 100:.2f}%",
                f"    {split_name} Frame Class F1  : {class_m['macro_f1']:.4f}",
                (
                    f"    {split_name} Frame Class AUC : {class_m['auc']:.4f}"
                    if not np.isnan(class_m["auc"])
                    else f"    {split_name} Frame Class AUC : NaN"
                ),
            ]
        )

    if video_class_m is not None:
        summary_lines.extend(
            [
                f"    {split_name} Video Class Acc : {video_class_m['acc'] * 100:.2f}%",
                f"    {split_name} Video Class B-A : {video_class_m['bal_acc'] * 100:.2f}%",
                f"    {split_name} Video Class F1  : {video_class_m['macro_f1']:.4f}",
                (
                    f"    {split_name} Video Class AUC : {video_class_m['auc']:.4f}"
                    if not np.isnan(video_class_m["auc"])
                    else f"    {split_name} Video Class AUC : NaN"
                ),
            ]
        )

    table_lines = []
    if concept_report is not None:
        table_lines = _format_concept_table(
            concept_report, f"Frame-Level Concept Metrics ({split_name})"
        )

    if video_concept_report is not None:
        table_lines.extend(
            _format_concept_table(
                video_concept_report, f"Video-Level Concept Metrics ({split_name})"
            )
        )

    all_output_lines = summary_lines + table_lines + [""]
    for line in all_output_lines:
        log_message(line, logger=logger, level="info")
