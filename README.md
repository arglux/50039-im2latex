# 50039-im2latex

Math formula to LaTeX

## How to Run

By using Python
1. Open command line or terminal
2. Run `pip install -r requirements.txt`
3. Run `app.py` using Python or run `python app.py` from command line

By running `.exe` (Windows Only):
1. Open `dist` folder
2. Double click `app.exe`

## Results

Some interesting results:

Perfectly transcribed:
![perfect](asset/result-1.png)

Transcribed with some mistakes:
![some-mistakes](asset/result-2.png)

Transcribed from a raw snipped image:
![from-raw-snippet](asset/result-4.png)

## Notes

1. The model is sensitive to size and therefore not all image formulas can be fed directly into the model. Some may work, some may not.
2. Additionally, due to limited time, training is only done on a subset of data; mostly short, single line equations
