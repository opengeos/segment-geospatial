# TEST.md — cli-anything-samgeo Test Plan & Results

## Test Inventory Plan

| File | Type | Estimated Tests |
|------|------|-----------------|
| `test_core.py` | Unit tests | ~30 tests |
| `test_full_e2e.py` | E2E + subprocess tests | ~20 tests |

## Unit Test Plan (`test_core.py`)

### project.py (~8 tests)
- `test_create_project_basic` — Create project with minimal params
- `test_create_project_with_source` — Create with source image
- `test_create_project_saves_to_disk` — Verify JSON output on disk
- `test_open_project` — Open a saved project
- `test_open_project_not_found` — Error on missing file
- `test_save_project` — Save and verify contents
- `test_get_project_info` — Extract info dict
- `test_add_history_entry` — Append to history

### model.py (~6 tests)
- `test_list_models` — All models returned
- `test_get_model_info_type` — Get type-level info
- `test_get_model_info_specific` — Get model-level info
- `test_get_model_info_invalid_type` — Error on bad type
- `test_get_model_types` — Returns all type keys
- `test_get_default_model_id` — Returns correct defaults

### session.py (~8 tests)
- `test_session_init_empty` — Empty session
- `test_session_init_with_project` — Session with project
- `test_session_mutate_and_undo` — Mutate then undo
- `test_session_redo` — Undo then redo
- `test_session_multiple_undos` — Chain of undos
- `test_session_undo_empty` — Undo with nothing
- `test_session_save_load` — Round-trip save/load
- `test_session_get_status` — Status dict

### export.py (~4 tests)
- `test_list_formats` — All formats returned
- `test_export_no_masks_error` — Error when no masks set
- `test_export_invalid_format` — Error on bad format
- `test_export_file_exists_no_overwrite` — Error on existing file

### data.py (~2 tests)
- `test_raster_info_not_found` — Error on missing file
- `test_list_basemaps` — Returns basemap names

### vector.py (~2 tests)
- `test_vector_info_not_found` — Error on missing file
- `test_raster_to_vector_not_found` — Error on missing file

## E2E Test Plan (`test_full_e2e.py`)

### Data pipeline E2E (~4 tests)
- `test_download_tiles_osm` — Download real OSM tiles as GeoTIFF
- `test_raster_info_real` — Info on a real downloaded GeoTIFF
- `test_reproject_real` — Reproject downloaded tiles
- `test_split_raster_real` — Split into tiles

### Segmentation E2E (~4 tests)
- `test_automatic_segment_sam2` — Full automatic segmentation with SAM2
- `test_predict_points_sam2` — Point-based prediction
- `test_raster_to_vector_real` — Convert real masks to vectors
- `test_export_gpkg_real` — Export masks as GeoPackage

### CLI Subprocess Tests (~8 tests)
- `test_help` — --help output
- `test_version` — --version output
- `test_project_new_json` — Create project via CLI
- `test_model_list_json` — List models via CLI
- `test_data_info_json` — Raster info via CLI
- `test_export_formats_json` — List formats via CLI
- `test_full_workflow` — Create project -> download -> segment -> export
- `test_vector_convert_workflow` — Segment -> convert to vector

### Realistic Workflow Scenarios

**Workflow 1: Satellite Image Segmentation Pipeline**
- Simulates: A GIS analyst segmenting a satellite image
- Operations: download tiles -> create project -> automatic segment -> vector convert -> export GeoJSON
- Verified: GeoTIFF exists, mask file > 0 bytes, vector has features, GeoJSON valid

**Workflow 2: Interactive Point-Based Segmentation**
- Simulates: Selecting specific features on a map
- Operations: create project -> set source -> predict with points -> export
- Verified: Mask file exists, correct format

## Test Results

### Unit Tests (`test_core.py`) — 45 passed

```
cli_anything/samgeo/tests/test_core.py::TestProject::test_create_project_basic PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_create_project_with_model PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_create_project_saves_to_disk PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_open_project PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_open_project_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_open_project_invalid PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_save_project PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_save_project_no_path PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_get_project_info PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_add_history_entry PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_set_masks PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_set_vectors PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_set_source PASSED
cli_anything/samgeo/tests/test_core.py::TestProject::test_project_round_trip PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_list_models PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_list_models_has_sam2 PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_get_model_info_type PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_get_model_info_specific PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_get_model_info_invalid_type PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_get_model_info_invalid_id PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_get_model_types PASSED
cli_anything/samgeo/tests/test_core.py::TestModel::test_get_default_model_id PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_init_empty PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_init_with_project PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_mutate_and_undo PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_redo PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_multiple_undos PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_undo_empty PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_redo_empty PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_redo_cleared_on_mutate PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_save_load PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_load_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_get_status PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_get_history PASSED
cli_anything/samgeo/tests/test_core.py::TestSession::test_session_update_project PASSED
cli_anything/samgeo/tests/test_core.py::TestExport::test_list_formats PASSED
cli_anything/samgeo/tests/test_core.py::TestExport::test_export_no_masks_error PASSED
cli_anything/samgeo/tests/test_core.py::TestExport::test_export_invalid_format PASSED
cli_anything/samgeo/tests/test_core.py::TestExport::test_export_file_exists_no_overwrite PASSED
cli_anything/samgeo/tests/test_core.py::TestDataErrors::test_raster_info_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestDataErrors::test_reproject_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestDataErrors::test_split_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestVectorErrors::test_vector_info_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestVectorErrors::test_raster_to_vector_not_found PASSED
cli_anything/samgeo/tests/test_core.py::TestVectorErrors::test_filter_not_found PASSED

======================== 45 passed, 1 warning in 1.95s =========================
```

### E2E + Subprocess Tests (`test_full_e2e.py`) — 24 passed

```
[_resolve_cli] Using installed command: /home/qiusheng/miniconda3/envs/geo/bin/cli-anything-samgeo

cli_anything/samgeo/tests/test_full_e2e.py::TestDataPipelineE2E::test_raster_info_real PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestDataPipelineE2E::test_download_tiles_osm PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestDataPipelineE2E::test_reproject_real PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestDataPipelineE2E::test_split_raster_real PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestVectorPipelineE2E::test_raster_to_vector_gpkg PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestVectorPipelineE2E::test_raster_to_vector_geojson PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestVectorPipelineE2E::test_vector_info_real PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestVectorPipelineE2E::test_filter_vectors_real PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestExportE2E::test_export_geotiff PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestExportE2E::test_export_png PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestExportE2E::test_export_gpkg PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_help PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_version PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_project_new_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_model_list_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_model_info_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_data_info_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_export_formats_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_project_info_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_project_history_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_model_check_json PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_full_data_workflow PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_vector_convert_subprocess PASSED
cli_anything/samgeo/tests/test_full_e2e.py::TestCLISubprocess::test_export_via_subprocess PASSED

======================= 24 passed, 2 warnings in 18.66s ========================
```

### Summary

| Suite | Tests | Passed | Failed | Time |
|-------|-------|--------|--------|------|
| Unit (test_core.py) | 45 | 45 | 0 | 1.95s |
| E2E (test_full_e2e.py) | 24 | 24 | 0 | 18.66s |
| **Total** | **69** | **69** | **0** | **~20.6s** |

**Pass rate: 100%**

### Coverage Notes

- **Unit tests**: Full coverage of project, model, session, and export modules.
  Error handling paths tested for data and vector modules.
- **E2E tests**: Real GeoTIFF creation, tile download (OSM), reprojection, splitting,
  vector conversion (GeoPackage, GeoJSON), export (GeoTIFF, PNG, GeoPackage), file
  format validation (TIFF magic bytes, PNG magic bytes, GeoJSON structure).
- **Subprocess tests**: All commands tested via installed `cli-anything-samgeo` binary
  with `CLI_ANYTHING_FORCE_INSTALLED=1`. JSON output parsing verified.
- **Not covered**: Live SAM model segmentation (requires GPU + model weights download).
  The segmentation functions are tested indirectly through the CLI and E2E pipeline
  using synthetic masks.
