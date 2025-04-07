import os
import sys
import subprocess


def main():
    if len(sys.argv) < 6:
        print("Usage: script.py <task> <bert> <domain> <run_dir> <runs> [cuda_devices]")
        sys.exit(1)

    task, bert, domain, run_dir, runs = sys.argv[1:6]
    runs = int(runs)
    cuda_devices = sys.argv[6] if len(sys.argv) > 6 else None

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        print(f"Using CUDA: {cuda_devices}")

    DATA_DIR = os.path.join("..", task, domain)
    print(DATA_DIR)

    for run in range(1, runs + 1):
        OUTPUT_DIR = os.path.join("..", "run_hsum", run_dir, domain, str(run))
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        valid_json = os.path.join(OUTPUT_DIR, "valid.json")
        predictions_json = os.path.join(OUTPUT_DIR, "predictions.json")
        model_path = os.path.join(OUTPUT_DIR, "model.pt")

        if not os.path.exists(valid_json):
            train_command = [
                "python", "../src/run_rrc_hsum.py",
                "--bert_model", bert, "--do_train", "--do_valid",
                "--gradient_accumulation_steps", "2",
                "--max_seq_length", "320", "--train_batch_size", "16",
                "--learning_rate", "3e-5", "--num_train_epochs", "4",
                "--output_dir", OUTPUT_DIR, "--data_dir", DATA_DIR,
                "--seed", str(run)
            ]
            with open(os.path.join(OUTPUT_DIR, "train_log.txt"), "w") as log:
                subprocess.run(train_command, stdout=log, stderr=log)

        if not os.path.exists(predictions_json):
            eval_command = [
                "python", "../src/run_rrc_hsum.py",
                "--bert_model", bert, "--do_eval", "--max_seq_length", "320",
                "--output_dir", OUTPUT_DIR, "--data_dir", DATA_DIR,
                "--seed", str(run)
            ]
            with open(os.path.join(OUTPUT_DIR, "test_log.txt"), "w") as log:
                subprocess.run(eval_command, stdout=log, stderr=log)

        if os.path.exists(predictions_json) and os.path.exists(model_path):
            os.remove(model_path)


if __name__ == "__main__":
    main()
