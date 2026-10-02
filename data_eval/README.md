Training evaluation uses the live model on rendered Objaverse scene folders.
It shares the scene loader, RGB sampler, and LPIPS/SSIM/PSNR implementation with
`inference/infer_batched.sh`. The launcher uses CFG 1, temperature 0.5, seed 0,
and no target-view padding.

Edit `eval_objaverse_small.json` to choose scenes and view pairs:

```json
{
  "test_paths": ["scene_folder_name"],
  "test_indices_src": [[0], [0, 3], [0, 3, 7]],
  "test_indices_tgt": [[1], [1, 2, 4], [1, 2, 4]]
}
```

Each `test_paths` entry must name a folder beneath the evaluation data root.
Existing full paths also work: their final folder names identify scenes.
View indices are zero-based positions in that scene's `transforms.json` frames
list. Eight-view scenes accept indices 0–7. Each source list is paired with the
target list at the same position; every pair runs on every selected scene.
The JSON controls the view combinations, including combinations with the same
counts but different indices.

The bundled JSON selects four scenes from the rendered eight-view data and
three source/target view combinations. Update it when using a different dataset.
Startup validates the scene manifests and indices before loading model weights.
Selected scene keys are also excluded from the normal training stream when
they match training scene keys.

Override the dataset, selection file, scene limit, or epoch frequency as needed:

```bash
EVAL_DATA_PATH=/path/to/rendered_objaverse \
EVAL_JSON=data_eval/eval_objaverse_small.json \
EVAL_MAX_SCENES=100 EVAL_FREQ=1 bash train_namvis.sh
```

`EVAL_MAX_SCENES` caps the selected scenes in JSON order. Evaluation runs before
each selected epoch, including epoch 0. The optional `--eval_iters_freq` training
argument enables evaluation between optimizer updates.

Metrics and evaluation settings are saved to
`<out_dir>/evaluation/<evaluation_tag>/metrics.json`. Per-pair metrics are logged
under `eval/objaverse/pairNNN_srcN_tgtM/`, alongside the existing aggregate keys.
The aggregate averages all evaluated target images. Epoch evaluations also
save `input/`, `gt/`, `gen/`, and comparison grids under each pair and scene;
evaluation between iterations saves metrics only.

The original TAR evaluation remains available with `--eval_backend=webdataset`
and a TAR path passed as `--eval_path`.
