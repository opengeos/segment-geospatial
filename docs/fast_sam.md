# FastSAM integration moved

FastSAM's geospatial integration is now part of the separately installed
`segment-anything-fast` distribution, which contains AGPL-covered code. SamGeo
no longer ships `samgeo.fast_sam` or installs FastSAM through `[all]` or `[extra]`.
The old `[fast]` extra has been removed.

For FastSAM workflows, use Python 3.10 or later and install the geospatial extra:

```bash
pip install --upgrade segment-geospatial "segment-anything-fast[geo]>=0.1.3"
```

Change the import in existing scripts and notebooks:

```python
from samgeo_fastsam import SamGeo

sam = SamGeo(model="FastSAM-x.pt")
sam.set_image("image.tif", device="cpu")
sam.everything_prompt(output="masks.tif")
```

The `SamGeo` methods, FastSAM-s/x model selection, point/box/text prompts and
mask/vector export behavior are retained. Existing environments must upgrade
`segment-anything-fast` to obtain the new import. The class's Python module path
changes; serialized objects referring to `samgeo.fast_sam` are not migrated.
Model checkpoints are still separate downloads; the code license is not a new
permission grant for those files.

These instructions require the coordinated releases containing the new FastSAM
module and the SamGeo removal. Older SamGeo releases still contain the adapted
wrapper; installing the new add-on alone does not remove it from an existing
environment. Upgrade SamGeo after its corrected release is available.

The agent CLI does not support this integration. Use the Python API above for
FastSAM, or select another supported model in the CLI. FastSAM's methods differ
from the `generate`/`predict` interface used by that CLI.

See the [FastSAM source and license notices](https://github.com/opengeos/FastSAM).
For a model whose publisher explicitly licenses both code and checkpoints under
Apache-2.0, see [SamGeo2](samgeo2.md). This separation addresses the FastSAM
integration; it is not a license audit of every other optional model.
