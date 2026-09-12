# User Guide & How to Run

## Complete End-to-End Run
```powershell
python main.py --video data/raw/videos/accident/mock_accident.mp4
```

## Testing on Normal Traffic
```powershell
python main.py --video data/raw/videos/normal/real_road_test.mp4
```

## Running Automated Tests
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
