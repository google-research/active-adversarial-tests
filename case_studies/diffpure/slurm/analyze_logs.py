import numpy as np
import glob
import argparse
import os


def parse_log(path):
    with open(path, "r") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]

    if len(lines) < 4:
        return None

    if lines[-4].startswith("interior-vs-boundary discimination"):
        asr = float(lines[-4].split(":")[1].strip())
        logit_diff = float(lines[-3].split(":")[1].split("+-")[0].strip())
        validation_acc = eval(lines[-2].split(":")[-1].replace("nan", "np.nan"))
        if type(validation_acc) is float:
            validation_acc = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        validation_acc = np.array(validation_acc)
        n_failed = int(lines[-1].split("for ")[1].split("/")[0].strip())

        return asr, logit_diff, validation_acc, n_failed
    else:
        return None


def main(input_folder):
    logs = glob.glob(os.path.join(input_folder, "*.log"))

    results = [(p, parse_log(p)) for p in logs]

    incomplete_logs = [it[0] for it in results if it[1] is None]
    if len(incomplete_logs) > 0:
        print("Found incomplete logs for experiments:")
        for it in incomplete_logs:
            print(f"\t{it}")

    results = [it[1] for it in results if it[1] is not None]

    if len(results) == 0:
        print("No results found.")
        return

    results = results[:512]

    properties = [np.array([it[i] for it in results]) for i in range(len(results[0]))]

    n_samples = len(results)
    n_failed_samples = np.sum(properties[3])

    # filter failed samples
    failed_samples = [idx for idx in range(len(properties[3])) if properties[3][idx] == 1]
    properties = [[prop[idx] for idx in range(len(prop)) if idx not in failed_samples] for prop in properties]

    import pdb; pdb.set_trace()

    means = [np.mean(prop, 0) for prop in properties]
    stds = [np.std(prop, 0) for prop in properties]

    print(f"ASR: {means[0]}")
    print(f"Normalized Logit-Difference-Improvement: {means[1]} +- {stds[1]}")
    print(f"Validation Accuracy (I, B, BS, BC, R. ASR S, R. ASR C): {tuple(means[2])}")
    print(f"Setup failed for {n_failed_samples}/{n_samples} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    args = parser.parse_args()
    main(args.input)

