# DSM-Guided Prompting for Mental Health Assessment

Python Code for DSM-Guided Large Language Model Reasoning for Depression and PTSD Assessment From Psychiatric Interview Transcripts (IEEE Access)

## Setup

Python 3.11.

    $ pip install -r requirements.txt
    $ cp .env.example .env      # then fill in your API keys

`OPENAI_API_KEY` and `DEEPSEEK_API_KEY` are read from the environment (or from `.env`).
No key is stored in this repository.

## Data

The DAIC-WOZ and E-DAIC corpora are **not** redistributed here — they are released under a
EULA and must be obtained from [DCAPS/USC-ICT](https://dcapswoz.ict.usc.edu/). Place the
downloaded corpora and all intermediate outputs under `data/`, which is gitignored:

    data/
    ├── e_daic_metadata/        # train_split.csv, test_split.csv
    ├── e_daic_text_only/       # <ID>_Transcript.csv, one per participant
    ├── results/                # per-experiment generation output
    ├── exp6/                   # experiment 6 train/test tables
    ├── prompts_exp6_training/  # one prompt .txt per Participant_ID
    ├── prompts_exp6_test/
    ├── checkpoints/            # long-run generation checkpoints
    └── eval/                   # predict_label*.xlsx for scoring

Every script's default paths point into `data/`. Scripts with a CLI accept `--input` /
`--output`; the rest expose the paths as constants at the top of the file or in `main()`.

`prompts/` holds the reusable prompt text: `system_prompt_template.txt` (with `{gender}`,
`{transcript}`, `{dsm5_criteria}` placeholders) and `dsm5_mdd_criteria.txt`. The
placeholders are there because the original prompt file embedded a participant transcript,
which the EULA does not permit us to publish.

## Experimental conditions

Notation: **P** = patient description (transcript, plus gender and DSM-5 criteria where
noted), **D** = diagnosis, **R** = medical rationale.

| Exp | Test-time input | LLM output | Intermediate step |
|-----|-----------------|------------|-------------------|
| 1 | Raw transcript | D | — |
| 2 | Raw transcript | summary feature | 30 generated on train (5 kept), 5 on test (1 kept) |
| 3 | P (+ gender + DSM-5) | D | — |
| 4 | P | summary feature | 30 generated on train (5 kept), 5 on test (1 kept) |
| 5 | P | R + D | train: P+D → 30 R (5 kept); test: 5 R+D (1 kept) |
| 6 | P + 2 shots from Exp 5 training data | R + D | few-shot, one depressed and one non-depressed exemplar |

Experiments 1–5 are driven by `exp1to5/`, selecting the condition with `--template`:

| `--template` | Condition |
|--------------|-----------|
| `template_for_experiment_2` | Exp 1 / 2 — transcript only |
| `gender_dsm` | Exp 3 — transcript + gender + DSM-5 → diagnosis |
| `template_for_experiment_4` | Exp 4 — summary with gender + DSM-5 |
| `template_for_experiment_4_with_ptsd` | Exp 4, PTSD criteria variant |
| `transcript_depression` | Exp 5 training — P+D → rationale |
| `main` | Full rationale generation with all fields |

All prompt templates live in `exp1to5/config.py` (`PromptTemplateConfig`).

## Pipeline

### 1. Build transcripts

Concatenate each participant's utterances into a single `Transcript` column and join with
the split metadata.

    $ python data_prep/make_transcript_edaic.py --type both    # E-DAIC (test, train, or both)
    $ python data_prep/make_transcript_daic.py --help          # DAIC-WOZ

Supporting utilities: `dedupe_by_participant_id.py` (one row per participant),
`filter_by_ids.py` (subset by an ID list), `check_token.py` (add a `Token_Count` column),
`to_excel.py` (CSV/Excel conversion and `PHQ_Binary` correction), `csv_merge.py`.

### 2. Generate rationales and summaries (Experiments 1–5)

    $ cd exp1to5
    $ python main.py --model gpt-4o --template gender_dsm --count 30 \
          --start_id 302 --end_id 718 --input <in.csv> --output <out.csv>

`--model` accepts `gpt-4o` or `deepseek-reasoner`. `--count` is the number of candidates
generated per participant (30 on the training side, 5 on the test side). Results are
written as `Rational_1 … Rational_n` columns, with periodic progress saves.

### 3. Select rationales

An LLM acting as a medical expert ranks the candidates and returns the best ones —
5 of 30 on the training side, 1 of 5 on the test side.

    $ python rationale_selection/rational_selector_gpt4o.py
    $ python rationale_selection/rational_selector_deepseek.py

Both expect a `Prompt` column alongside `Rational_*`; build it with
`rationale_selection/make_only_prompt.py` if it is missing. Then expand the selected
numbers back into text and clean the output:

    $ python rationale_selection/make_selected_rationale.py
    $ python rationale_selection/exp5_selected_rationale.py
    $ python rationale_selection/postprocessing.py

`postprocessing.py` strips bold markdown and any trailing `Diagnosis:` span, writing a
`modified_rationale` column — this prevents the label from leaking into the rationale used
downstream.

### 4. Experiment 6 (few-shot)

Run in order; each step consumes the previous step's output.

    $ python exp6/exp6_training_shot_selection_from_exp5_output.py
    $ python exp6/exp6_training_make_prompt.py
    $ python exp6/exp6_train_generation_from_prompt_files.py
    $ python exp6/exp6_test_shot_selection_from_exp6_training_output.py
    $ python exp6/exp6_test_make_prompt.py
    $ python exp6/exp6_test_generation_from_prompt_files.py

Shot selection draws one depressed and one non-depressed exemplar per target under a fixed
RNG seed, so the assignment is reproducible: seed `42` on the training side, seed
`20250508` on the test side, where exemplars are additionally drawn without reuse.
The `*_make_prompt.py` steps write one prompt `.txt` per `Participant_ID`; the
`*_generation_*.py` steps read that directory and call the API with checkpointing —
pass `--resume <checkpoint.xlsx>` to continue an interrupted run.

### 5. Evaluate

Scoring reads an Excel file with a `label` column and one column per model/condition
(`predict1 … predictN`), maps the textual labels to 0/1, and reports TP/FP/FN/TN with
per-class and macro F1.

    $ python evaluation/eval_metrics.py           # E-DAIC, depression
    $ python evaluation/eval_metrics_daic.py      # DAIC-WOZ, depression
    $ python evaluation/eval_metrics_ptsd.py      # PTSD, with sensitivity/specificity

### Single call

A minimal one-shot check that the API and prompt wiring work:

    $ python examples/single_call_gpt4o.py --prompt my_system_prompt.txt

## Repository layout

    data_prep/           transcript construction and table utilities
    exp1to5/             Experiments 1–5 generation engine (config/models/utils/main)
    rationale_selection/ LLM-based candidate selection and post-processing
    exp6/                few-shot pipeline, in execution order
    evaluation/          classification metrics
    prompts/             prompt template and DSM-5 criteria text
    examples/            minimal single-call example

## Generation settings

`temperature = 1.0`, `max_tokens = 500` for rationale and summary generation;
`max_tokens = 100` for the selection step. Models: `gpt-4o` and `deepseek-chat`
(DeepSeek-V3). Because generation is sampled at temperature 1.0, regenerated rationales
will not match the published ones token for token; the reported metrics come from the
prediction files produced by the runs described above.

## Citation

    @article{,
      title   = {},
      author  = {},
      journal = {},
      year    = {},
    }
