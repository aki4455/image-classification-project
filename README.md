# Car brand and colour classifier for Tashkent street photos

CSAI-3210 midterm. Give it a phone photo; it prints the brand and colour of every car in it:

```
=== photo_29.jpg  (3 car(s), 0.1s)
  1. Zeekr  p=0.38 (next: BMW 0.10) | colour black | det 0.89 | box 618x479
  2. Chevrolet (Tracker 0.87)  p=0.84 (next: Omoda 0.03) | colour black | det 0.90 | box 586x436
  3. Chevrolet (Onix 0.63)  p=0.62 (next: Omoda 0.10) | colour black | det 0.79 | box 221x400
ANSWER: Zeekr black; Chevrolet black; Chevrolet black
```

Pipeline: YOLO11 (COCO, pretrained) finds the cars -> each car is cropped -> a fine-tuned ConvNeXt-Tiny
names the brand (21 brands) and, for Uzbek Chevrolets, the model -> a colour model (frozen ResNet-18
features + linear layer, with HSV hue rules for red/green/yellow) names the colour (8 colours).
Everything runs locally; nothing calls the internet at prediction time. A web UI shows the same results
in the browser, for uploaded photos or live from a camera (section 2).

## 1. Install (clean machine, Windows or Linux)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and git.
2. Get the code, the weights and the environment (Python 3.11; PyTorch comes from the PyTorch CUDA 13.0 index,
   it also runs on a machine without an NVIDIA GPU, only slower). The brand network (111 MB) is stored with
   Git LFS, which comes with Git for Windows (elsewhere install git-lfs and run `git lfs install` once):
   ```
   git clone https://github.com/aki4455/image-classification-project.git
   cd image-classification-project
   git lfs pull
   uv sync
   ```
3. The weights are in the repository:
   | File | What it is |
   | --- | --- |
   | `runs/v2_two_head/best.pt` | brand + car-model network (Git LFS) |
   | `weights/yolo11n.pt` | YOLO11 detector, COCO-pretrained (also at https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) |
   | `inference/colour_model.pt`, `inference/colour_params.json` | colour model + thresholds |

   If `best.pt` is only a few hundred bytes after cloning, Git LFS was missing: install it and run `git lfs pull`.
4. Stop the detector library from sending usage statistics (once):
   ```
   uv run yolo settings sync=False
   ```
5. Check that it works offline: turn Wi-Fi off and run one photo (section 2).

Alternative without uv: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130`.

## 2. Run (test day)

All commands run from the repository folder.

Watch a folder; every new photo in it is answered automatically:
```
uv run python -m inference.predict --watch "C:\Users\user\Downloads\Telegram Desktop"
```
One or more photo files:
```
uv run python -m inference.predict photo1.jpg photo2.jpg
```
Copy the `ANSWER:` line onto the answer sheet. `MODELS:` adds the Chevrolet model and is not scored.
Annotated copies of each photo are saved to `outputs/`.

Useful options: `--weights PATH` (another brand network), `--min-conf 0.4` (detector confidence needed),
`--min-rel 0.15` (ignore cars smaller than this share of the largest car's box), `--no-flip` (classify each
car once instead of averaging it with its mirror image; this is the pipeline behind the results in section 5).

### Web UI: photos, live camera, model overview

```
uv run python -m web.server
```
This opens http://127.0.0.1:8765 in the browser. It runs the same pipeline as the command line, on the same defaults.
- **Photos**: drop, paste or upload photos, several at once. Every car gets a numbered box with its brand,
  Chevrolet model, colour and confidence, plus the crop the classifier saw and the runner-up brands. The
  `ANSWER` line has a copy button, and "Export CSV" saves one row per car for all analysed photos.
- **Live camera**: predictions are drawn on the webcam picture. "Smooth over frames" follows each car and
  averages its probabilities over the last frames; "Freeze & analyse" sends the current frame to the Photos tab.
- **About the model**: the pipeline, validation results, training curves and confusion matrices of the loaded weights.

The settings button (top right) has the same options as the command line. `--samples "D:\test_car_photos"`
lists a folder of photos as one-click samples; `--weights`, `--port` and `--no-browser` also work.
The page loads nothing from the internet. The camera works when the page is opened on this computer or through
the `--public` link (below), because browsers allow cameras only on localhost or HTTPS. With `--host 0.0.0.0`, a
phone on the same Wi-Fi can open `http://<this computer's IP>:8765` and upload photos; its "Take photo" button
opens the phone's camera.

For scripts: `curl -F "file=@photo.jpg" http://127.0.0.1:8765/api/predict` returns every car as JSON;
http://127.0.0.1:8765/docs lists the options.

### Public link (Tailscale Funnel): run at home, show anywhere

`--public` also puts the page on the internet while the server runs, at an HTTPS address that stays the same:
`https://<computer name>.<tailnet name>.ts.net`. The models keep running on this computer's GPU; the link opens
in any browser, phones included, and the live camera works through it. It uses
[Tailscale Funnel](https://tailscale.com/kb/1223/funnel): free, no warning page, nothing to install for visitors.

One-time setup, a day ahead (a new address can take up to 10 minutes to start working):
1. Install Tailscale from https://tailscale.com/download/windows and sign in (Google, Microsoft or GitHub account).
2. Open a new terminal and run `tailscale funnel 8765`. The first time, it prints links that turn on HTTPS and
   Funnel for your account: open them and approve. When it prints `Available on the internet:` and the address,
   press Ctrl+C.

On the day:
```
uv run python -m web.server --samples "D:\test_car_photos" --public
```
It prints `public link: https://...`. The link works while that window is open and the computer is on and online;
Ctrl+C closes it together with the server. While it runs, Windows does not go to sleep (a laptop still sleeps
when its lid is closed). Anyone with the link can use the page, so stop the server after the demo. Before you
leave home, open the link on your phone with Wi-Fi off, and pause Windows Update for a week (Settings, Windows
Update) so it does not restart the computer.

## 3. Reproduce the training

Data:
- **Car-1000** (Hu et al., 2025): https://github.com/toggle1995/Car-1000 - download and unpack, e.g. to `D:\Car-1000-dataset`.
- **Our own photos** (403 Uzbek Chevrolet listing photos): <PRIVATE LINK, shared with the instructor>.
  They are photos posted by private sellers, so we do not publish them openly.
- Labels in this repository: `labels_uzbek.csv` (our brand + colour labels), `brand_map.csv`
  (Car-1000 model folder -> badge), `selected_images.txt` (the Car-1000 images we used).

Steps, in order (paths are examples):
```
uv run python -m preprocessing.clean_artifacts "D:\Car-1000-dataset"                     # dry run: files not in the dataset
uv run python -m preprocessing.build_brand_map "D:\Car-1000-dataset" --cap 1000          # brand_map.csv, selected_images.txt
uv run python -m utils.crop_cars "D:\Car-1000-dataset\all_images" "D:\crops\car1000_v2" largest --min-frac 0.1 --only selected_images.txt
uv run python -m utils.crop_cars "D:\raw\uzbek_cars" "D:\crops\uzbek_cars_cropped" largest --min-frac 0.1
uv run python -m data.labels "D:\crops\uzbek_cars_cropped" labels_uzbek.csv --source web-listings --default-brand Chevrolet
uv run python -m data.check_data "D:\crops\car1000_v2" brand_map.csv "D:\Car-1000-dataset\dataset_split\dataset_split.json" labels_uzbek.csv
uv run python -m train.train --car1000 "D:\crops\car1000_v2" --brand-map brand_map.csv --split "D:\Car-1000-dataset\dataset_split\dataset_split.json" --uzbek labels_uzbek.csv --out runs\v2_two_head
uv run python -m train.fit_colour_model labels_uzbek.csv                                  # colour: rules vs learned vs hybrid
```
Training the brand network takes about 7 minutes on an RTX 3070 Ti.

## 4. Repository layout

| Folder | Contents |
| --- | --- |
| `utils/` | detector loading, cropping, dataset search helpers |
| `preprocessing/` | brand mapping and sampling, dataset clean-up |
| `data/` | dataset and transforms, labelling tool, data check |
| `train/` | network definition, brand training, colour fitting |
| `inference/` | `predict.py` (test-day pipeline), colour model |
| `web/` | `server.py` (web UI and JSON API), `static/` (the page: HTML, CSS, JavaScript) |

## 5. Results (details in the report)

| Measure | Result |
| --- | --- |
| Brand accuracy, Car-1000 validation / test | 96.3% / 96.7% (21 brands) |
| Brand accuracy, our Chevrolet validation photos | 96.3% (81 photos) |
| Colour accuracy, 5-fold cross-validation | 82.6% (rules alone: 77.2%) |
| Own Tashkent phone photos, exam scoring | 18.3 of 25 points |
| Brand accuracy with mirror averaging, Car-1000 validation / test | 96.6% / 97.1% (our Chevrolet photos: 96.3%, unchanged) |

Mirror averaging (classify each crop and its mirror image, average the probabilities) was measured after
training and is now the default. It has nothing to tune, so the test-set gain (119 -> 103 errors out of 3,569) is
not fitted to the test set. All other numbers in this table, and the example output at the top, come from
the single pass that `--no-flip` still gives.

AI use: built with help from Claude (Anthropic) as tutor and pair programmer; see the report's AI-use statement.