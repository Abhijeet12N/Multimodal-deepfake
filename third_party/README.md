# Third-party dependencies

## AASIST

`aasist/` is a Git submodule from [clovaai/aasist](https://github.com/clovaai/aasist), pinned to tested commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`.

Initialize it in an existing clone:

```powershell
git submodule update --init --recursive
```

For a new clone, use `git clone --recurse-submodules` as shown in the main README. The parent repository stores the upstream URL and commit reference. The upstream MIT `LICENSE` and `NOTICE` remain in the submodule; its source is unmodified.

The pinned upstream revision already includes the small official AASIST and AASIST-L pretrained checkpoints. Those are upstream assets, not newly uploaded project training files. Project fine-tuned checkpoints stay in `models/` and are ignored by Git. Share those separately using an artifact host, Drive, Hugging Face, or a release download; datasets and training caches are also excluded.

Runtime source lookup prefers this submodule. Older `aasist/` and sibling `../aasist` layouts remain supported for existing workspaces and previously exported Colab bundles. Source loading uses the selected `models/AASIST.py` directly to avoid conflicts with unrelated Python packages named `models`.

Do not advance the submodule to another upstream commit without checking checkpoint compatibility and inference parity. The Colab bundle generator packages the selected source, both official initial checkpoints, `LICENSE`, and `NOTICE` under `Multimodal-deepfake/third_party/aasist/`; it does not package Git history, local fine-tuned exports, `.env`, or evaluation recordings.
