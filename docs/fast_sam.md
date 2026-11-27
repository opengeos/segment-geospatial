# FastSAM integration moved

FastSAM's geospatial integration is now part of the separately installed
`segment-anything-fast` distribution, which contains AGPL-covered code. SamGeo
no longer ships `samgeo.fast_sam` or installs FastSAM through `[all]` or `[extra]`.
The old `[fast]` extra has been removed.

For FastSAM workflows, use Python 3.10 or later and install the geospatial extra:

```bash
pip install --upgrade segment-geospatial "segment-anything-fast[geo]>=0.1.3" "setuptools<81"
```

Change the import in existing scripts and notebooks:

```python
from samgeo_fastsam import SamGeo

sam = SamGeo(model="FastSAM-x.pt")
sam.set_image("image.tif", device="cpu")
sam.everything_prompt(output="masks.tif")
```

FastSAM depends on an older `ultralytics` release. That release imports
`pkg_resources`, which setuptools 81 and later no longer provide, hence the
`setuptools<81` above. On torch 2.6 or later, loading `FastSAM-x.pt` also fails
with `UnpicklingError: Weights only load failed` because `torch.load` now
defaults to `weights_only=True`. Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before
starting Python to load it. This turns off the weights-only safety check, so use
it only with checkpoints you trust, such as the official FastSAM weights. Both
limitations are tracked in
[#580](https://github.com/opengeos/segment-geospatial/issues/580).

The `SamGeo` methods, FastSAM-s/x model selection, point/box/text prompts and
mask/vector export behavior are retained. Existing environments must upgrade
`segment-anything-fast` to obtain the new import. The class's Python module path
changes; serialized objects referring to `samgeo.fast_sam` are not migrated.
Model checkpoints are still separate downloads; the code license is not a new
permission grant for those files.

The new FastSAM module ships in `segment-anything-fast` 0.1.3 and later. SamGeo
removes its adapted wrapper in 1.5.0; SamGeo 1.4.2 and earlier still contain
`samgeo.fast_sam`, and installing the add-on alone does not remove it from an
existing environment, so upgrade SamGeo as well.

The agent CLI does not support this integration. Use the Python API above for
FastSAM, or select another supported model in the CLI. FastSAM's methods differ
from the `generate`/`predict` interface used by that CLI.

See the [FastSAM source and license notices](https://github.com/opengeos/FastSAM).
For a model whose publisher explicitly licenses both code and checkpoints under
Apache-2.0, see [SamGeo2](samgeo2.md). This separation addresses the FastSAM
integration; it is not a license audit of every other optional model.
