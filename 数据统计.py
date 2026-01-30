from pathlib import Path

from pathlib import Path
import json


if __name__ == '__main__':

    from pathlib import Path
    import json

    folder = r'results'

    print("层数\tPrecision\tRecall\tF1\tmAP50\tmAP50-95\tTime(s)")

    for file in Path(folder).rglob('*.json'):
        data = json.load(open(file, 'r', encoding='utf-8'))
        p = data["metrics/precision(B)"]
        r = data["metrics/recall(B)"]
        m50 = data["metrics/mAP50(B)"]
        m95 = data["metrics/mAP50-95(B)"]
        t = data["count_time_seconds"]
        f1 = 2 * p * r / (p + r)

        print(f"{file.stem}\t{p:.6f}\t{r:.6f}\t{f1:.6f}\t{m50:.6f}\t{m95:.6f}\t{t}")

