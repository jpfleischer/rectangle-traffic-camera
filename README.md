```bash
python download_ortho.py
python crop_ortho_interactive.py
python pair_points.py
```

to be used in conjunction with <https://github.com/jpfleischer/TrackerMALT/>
copy .env.example to .env and set the correct values.

```bash
python visualize_tracks_gui.py --tif ortho_zoom.tif
```

find out how many people are braking from 15, 20 meters, etc.

--------------------------------


```bash
DBG_VIDEO=hiv00425.mp4 DBG_TRACK=746 python braking_metrics.py   --intersection-id 1   --approach-id toward_cam_main    --divider-side negative   --min-entry-speed 0.5   --min-delta-v 0.6   --accel-trigger 0.25   --mild-g 0.15   --moderate-g 0.25   --severe-g 0.40   --smooth 3   -v --write-events 
```
