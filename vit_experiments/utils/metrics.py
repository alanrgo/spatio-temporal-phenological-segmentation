import os
import json
from datetime import datetime


def _round_dict(d, ndigits=6):
	def _round(v):
		try:
			return round(float(v), ndigits)
		except Exception:
			return v
	return {k: _round(v) for k, v in d.items()}


def save_metrics(output_path, final, best, overall):
	"""Save metrics in a human-readable manner.

	Files written to `output_path`:
	- metrics_final.json
	- metrics_best_val.json
	- metrics_best_test.json
	- metrics_summary.txt
	"""

	os.makedirs(output_path, exist_ok=True)

	final_r = _round_dict(final)
	best_r = _round_dict(best)
	overall_r = _round_dict(overall)

	# Human-readable text summary
	summary_path = os.path.join(output_path, "metrics_summary.txt")
	lines = [
		f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
		"",
		"[Final Accuracies]",
		f"  train_acc: {final_r.get('train_acc')}",
		f"  val_acc:   {final_r.get('val_acc')}",
		f"  test_acc:  {final_r.get('test_acc')}",
		"",
		"[Best (by Val Acc)]",
		f"  train_acc: {best_r.get('train_acc')}",
		f"  val_acc:   {best_r.get('val_acc')}",
		f"  test_acc:  {best_r.get('test_acc')}",
		"",
		"[Best (by Test Acc)]",
		f"  train_acc: {overall_r.get('train_acc')}",
		f"  val_acc:   {overall_r.get('val_acc')}",
		f"  test_acc:  {overall_r.get('test_acc')}",
		"",
	]
	with open(summary_path, "w") as f:
		f.write("\n".join(lines))

	return {
		"summary": summary_path,
	}
