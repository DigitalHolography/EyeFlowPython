# EyeFlowPython

## Purpose

EyeFlowPython is a clean Python rewrite of EyeFlow focused on downstream analysis.

Important scope change:

- EyeFlowPython does not run AI segmentation.
- Vessel masks are expected to already exist in the input H5 files.
- The application consumes precomputed data and produces biomarkers, quality-control outputs, and exported analysis results.

## Product Goals

- Input one H5 file, a folder tree of H5 files, or a zip archive containing H5 files.
- Output one analysis result per input file.
- Preserve the input folder structure in batch and zip processing.
- Provide a CLI and a GUI with the same analysis capabilities.
- Keep the GUI user friendly and oriented around review, configuration, execution, and quality control.
- Produce clean, structured, reproducible outputs.

## Launching The App

### Install

Create and activate a virtual environment, then install the project.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

If you also want the Streamlit GUI:

```powershell
pip install -e .[gui]
```

### Launch The CLI

Validate one H5 file, a folder tree, or a zip archive:

```powershell
eyeflow validate path\to\input.h5
eyeflow validate path\to\folder
eyeflow validate path\to\archive.zip
```

You can also use the module form:

```powershell
python -m eyeflow validate path\to\input.h5
```

### Launch The GUI

Once the GUI dependencies are installed:

```powershell
eyeflow-gui
```

Fallback direct Streamlit command:

```powershell
streamlit run src\eyeflow\gui\app.py
```

## Scope

### In Scope

- Loading H5 files containing:
  - statistical moments
  - precomputed masks
  - acquisition metadata
  - optional spectral data
- Preprocessing before analysis.
- Velocity and waveform analysis.
- Pulse-wave velocity analysis.
- Cross-section signal generation.
- Cross-section result export and derived hemodynamic metrics.
- Spectral analysis when the required spectral data is present.
- Batch processing from folders and zip archives.
- Persistent user settings for GUI and CLI.
- H5, JSON, and log outputs.
- CPU-first implementation with optional GPU acceleration where useful.

### Out of Scope

- AI inference inside EyeFlowPython.
- Training or shipping segmentation models.
- Automatic generation of artery and vein masks from raw images.
- Manual mask painting inside the application.
- Full MATLAB visual/report parity for the first Python release.

The GUI may still provide mask visualization and overlay-based quality control, but not mask creation.

## Input Contract

The data contract must be strict and documented early. A clean Python implementation depends more on a stable input schema than on UI details.

### Required Datasets

- `/moment0`
  - 3D array: `(height, width, frames)`
- `/moment1`
  - 3D array: `(height, width, frames)`
- `/moment2`
  - 3D array: `(height, width, frames)`
- `/masks/artery`
  - 2D binary mask: `(height, width)`
- `/masks/vein`
  - 2D binary mask: `(height, width)`

### Optional Datasets

- `/SH`
  - Required only if spectral analysis is enabled.
- Additional masks if useful for QC or downstream analysis
  - examples: vessel, background, diaphragm, optic-disc, cross-section exclusions

### Required Metadata

At least one valid way to recover the time axis and one valid way to recover the spatial scale must be present.

- Frame timing metadata
  - preferred: timestamps in microseconds
  - fallback: frame rate and stride
- Spatial calibration metadata
  - preferred: `pixel_size_mm`
  - fallback: another explicit upstream calibration value already resolved before EyeFlowPython
- Frame range metadata if the source data has already been cropped upstream
- Any metadata needed to interpret units correctly

### Strong Recommendation

Do not make EyeFlowPython infer physical units from weak assumptions. If a biomarker depends on time or spatial scale, the needed metadata should be explicitly present in the input.

## Analysis Pipeline

The application should be structured as a dependency-driven pipeline.

### Stage 1: Load And Validate

- Validate required datasets and metadata.
- Validate shapes, dtypes, mask consistency, and units.
- Reject malformed inputs early with clear errors.

### Stage 2: Preprocess

Preprocessing remains part of the Python app even if masks are already provided.

- frame cropping
- rigid registration
- local normalization
- resizing
- interpolation
- outlier handling
- optional non-rigid registration if still needed

### Stage 3: Core Analysis

- blood-flow velocity analysis
- arterial and venous waveform analysis
- heartbeat-related metrics
- per-beat and statistical metrics

### Stage 4: Advanced Analysis

- pulse-wave velocity
- cross-section signal generation
- cross-section exports
- flow-rate and hemodynamic outputs
- spectral analysis when `SH` is available

## Dependency Rules

Some metrics depend on earlier stages. This should be implemented explicitly as a dependency graph, not as scattered conditionals.

Examples:

- preprocessing -> velocity analysis
- velocity analysis -> waveform biomarkers
- velocity analysis -> pulse-wave velocity
- velocity analysis + masks + spatial calibration -> cross-section analysis
- cross-section analysis -> exported cross-section biomarkers
- `SH` + masks + timing metadata -> spectral analysis

The user must be able to select requested outputs, and the app should automatically execute the required upstream dependencies.

## Outputs

Each processed input should generate a clean result package.

### Required Outputs

- one result H5 file with structured biomarker datasets
- one result JSON file for scalar and summary outputs
- one execution log file
- one saved copy of the effective analysis settings used for the run

### Output Requirements

- stable dataset naming
- explicit units
- quality-control fields
- run metadata
- clear failure reporting for partial or skipped modules

Optional figures and visual exports can be added later, but they are not the core deliverable for v1.

## CLI And GUI

The CLI and GUI must expose the same analysis features.

### CLI Requirements

- single-file processing
- folder-tree batch processing
- zip processing
- settings file support through `eyeflow-settings.json`
- module selection
- clear logging and non-zero exit codes on failure

### GUI Requirements

- built with Streamlit
- same processing options as the CLI
- input browsing
- batch execution
- settings persistence
- progress reporting
- visualization of loaded moments and masks
- quality-control views for overlays and key intermediate outputs

The GUI should be a review and execution interface, not a second implementation of the pipeline.

## Technical Requirements

- Python project managed with a `pyproject.toml`
- easy setup with a virtual environment
- use standard scientific Python libraries such as `numpy`, `h5py`, and related tooling
- modular codebase with small focused files
- clear separation between:
  - IO
  - validation
  - preprocessing
  - analysis modules
  - exports
  - CLI
  - GUI
- optional GPU support designed in from the start, but not required for all modules

## Architecture Principles

- The data contract comes first.
- The pipeline must be deterministic and testable.
- Every module should have a clear input and output interface.
- Failures should be local when possible: one failed optional module should not necessarily invalidate the whole run.
- Re-running downstream modules should not require repeating expensive upstream work when cached intermediates are available.
- The codebase should stay readable and easy to extend.

## Suggested V1 Deliverable

A good first Python release would include:

- strict H5 input validation
- preprocessing
- velocity analysis
- waveform metrics
- cross-section analysis
- H5 and JSON export
- CLI and Streamlit GUI parity
- settings persistence
- batch and zip support

Then add, in later iterations:

- pulse-wave velocity
- spectral analysis
- richer QC exports
- more visual reports

## Summary

EyeFlowPython is an analysis application, not a segmentation application.

Its success depends on:

- a strict H5 input schema
- explicit metadata for time and spatial calibration
- a clean dependency-aware analysis pipeline
- consistent outputs across CLI and GUI
