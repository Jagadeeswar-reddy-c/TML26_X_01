# TML26 Assignment 01 - Membership Inference Attack - Team X

This repository contains my final code for the Membership Inference Attack task in Trustworthy Machine Learning.

## Best Public Leaderboard Result

```text
Public leaderboard score: 0.057020
Method: Confidence-based Membership Inference Attack with public-set calibration
````

## Required Files

Before running the code, keep these files in the same folder:

| File | Link |
| :--- | :--- |
| **MIA.py** | (Local Script) |
| **pub.pt** | [View/Download File](https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt) |
| **priv.pt** | [View/Download File](https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt) |
| **model.pt** | [View/Download File](https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt) |
| **requirements.txt** | (Local Dependencies) |


## Install Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Run the Attack

### Windows

```bash
python MIA.py
```

### macOS / Linux

```bash
python3 MIA.py
```

## Output

After running the script, it will create:

```text
submission.csv
```

This file contains two columns:

```text
id,score
```

The script also submits `submission.csv` to the evaluation server.

## Notes

The code automatically selects the available device:

```text
Apple MPS -> CUDA -> CPU
```

So it can run on macOS, Windows, or Linux depending on the available hardware.

## requirements.txt

```txt
torch
torchvision
pandas
requests
```

