# Sample videos

Place sample videos in this folder to list them on the demo page. Supported formats are
`.mp4`, `.avi`, `.mov`, and `.mkv`. The app also allows direct video uploads, so samples are
optional.

> FakeAVCeleb v1.2 clips are not included. The dataset is licensed for research use and must
> not be redistributed. It also contains deepfakes of real people. The Git ignore rules exclude
> video clips from this folder.

## Optional ground-truth labels

The app infers a sample's ground-truth category from its filename. Include one of these strings
to enable comparison with the model output:

| Filename contains | Category | Video | Audio |
|---|---|---|---|
| `RealVideo-RealAudio` | R-R | real | real |
| `FakeVideo-RealAudio` | F-R | fake | real |
| `RealVideo-FakeAudio` | R-F | real | fake |
| `FakeVideo-FakeAudio` | F-F | fake | fake |

Example: `RealVideo-FakeAudio_sample1.mp4`. Without one of these names, the video is analyzed
without a ground-truth comparison.
