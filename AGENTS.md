# Instructions for AI Agents

This file guides any agent working in this repository - Codex, Claude, Copilot, or
another tool. Follow these rules before creating, modifying, or removing files.

## 1. Project Context

This is **Tech Challenge - Phase 3**, focused on prediction and analytical
intelligence for literacy in Brazil.

The goal is to create a Machine Learning solution that estimates the risk of a
student being classified as literate or not literate, using educational, geographic,
and socioeconomic data. The solution must produce interpretable and useful analyses
for public managers, not automated decisions about people.

Read [README.md](README.md) and [plan.md](plan.md) before starting substantial work.
The plan documents the architecture, stages, and acceptance criteria.

## 2. Mandatory Data and Modeling Principles

- The target variable is `alfabetizado` (0 or 1).
- **Never use `proficiencia` as an input feature.** In Phase 2 it was used to derive
  `alfabetizado`; using it during training creates data leakage.
- Do not use columns that are available only after the assessment, nor aggregations
  from the same year that include the student being predicted.
- Define and document the prediction point in time. As a default rule, to predict
  year `t`, use data known before the assessment in `t` and historical indicators
  from `t-1` or earlier.
- Student, school, and municipality IDs may be used for joins, auditing, and data
  splitting, but must not be used directly in the model.
- Treat student data as sensitive: do not publish microdata, identifiers,
  individual predictions, or credentials.
- Distinguish statistical association from causality in every analysis and
  presentation.

## 3. Intended Architecture

Phase 2 already uses S3, AWS Glue, Glue Data Catalog, and Athena, with
Bronze/Silver/Gold layers. In this phase, the ML reference will be a new Gold table:

`ml_dataset_alunos`

It will contain one row per student and year, validated features, and the target. It
will be built from the Phase 2 Silver layer, existing Gold tables, and, only when
justified, public external sources with reliable municipality keys and time periods.

Expected flow:

~~~text
Silver / external sources
        -> feature job (Glue/PySpark)
        -> Gold ml_dataset_alunos in Parquet
        -> quality validation and Athena catalog
        -> EDA, training, and evaluation in Python
        -> artifacts, report, and aggregated risk ranking
~~~

## 4. Code Organization

Keep this organization when adding implementation:

~~~text
data/             # small samples; complete data must not enter Git
notebooks/        # EDA and narrative; no exclusive critical logic
src/
  data/           # reading, contracts, and validation
  features/       # feature creation and the ML Gold layer
  preprocessing/  # reusable transformers
  modeling/       # training, parameter search, and prediction
  evaluation/     # metrics and explainability
  visualization/  # charts and executive tables
  aws/            # Glue jobs and AWS integration
tests/
  unit/
  integration/
reports/
  figures/
infra/            # Terraform or AWS SAM, if adopted
~~~

Use notebooks to explore and communicate. Move any logic needed for reproduction,
testing, or cloud execution into `src/`.

## 5. Implementation Standards

- Use Python 3.12 and keep dependencies in `requirements.txt`.
- Although Python has dynamic typing, explicitly type every variable, parameter, and
  function return so that types are always clear.
- Use `scikit-learn Pipeline` and `ColumnTransformer` so that imputation,
  transformations, and the model are fitted only on training data.
- Start with a simple baseline and compare it with more expressive models.
  Suggested options: DummyClassifier, Logistic Regression, and HistGradientBoosting
  or Random Forest.
- Preserve a final test set. Prefer temporal validation; if there are not enough
  years, use validation grouped by municipality or school and record the limitation.
- Evaluate more than accuracy: confusion matrix, precision, recall, F1, and, when
  appropriate, ROC-AUC and PR-AUC. Pay special attention to recall for the not
  literate class.
- Record random seeds, data versions, parameters, and metrics.
- Produce explainability with permutation importance and SHAP when feasible.
- Write small, typed functions when appropriate, with clear error messages and no
  secret values in the code.

## 6. Quality Before Completing a Change

1. Inspect the impact on existing files before editing them.
2. Do not overwrite or revert another person/agent's changes without authorization.
3. Add or update tests for new rules in `src/`.
4. Run the applicable checks:
   - `ruff check .`
   - `pytest`
5. Update the README, plan, or documentation when changing architecture, usage
   commands, data contracts, metrics, or results.
6. Clearly report what changed, how it was verified, and any limitations.

If production data is not available, use a synthetic or anonymized sample for
development and testing only. Explicitly state assumptions about schemas, temporal
coverage, and data quality.

## 7. AWS, Security, and Costs

- Never commit AWS keys, tokens, `.env` files, credentials, or confidential data.
- Use IAM roles, Secrets Manager, and OIDC in CI when external integration is needed.
- Prefer Parquet and year-based partitioning for analytical data.
- Before creating paid resources, describe the impact and confirm that creation is
  within the requested scope.
- Prioritize local execution and batch jobs. An online endpoint, dashboard, and
  multiple external sources are future enhancements, not challenge prerequisites.

## 8. Git and Collaboration

- Work with descriptive branches, for example:
  `feat/ml-dataset`, `feat/model-baseline`, or `docs/readme`.
- Make small, focused commits, using messages such as
  `feat: create preprocessing pipeline` or
  `test: cover ML dataset validation`.
- Do not force-push, perform destructive resets, or change remote configuration
  without explicit approval.
- Open pull requests with a summary, executed tests, risks, and evidence when the
  team workflow is active.

## 9. Expected Deliverables

The final delivery must contain:

- organized and reproducible code;
- a Gold feature table for ML;
- exploratory analysis and visualizations;
- a preprocessing, training, and validation pipeline;
- evaluation, interpretability, and limitations;
- risk ranking/aggregation for municipalities, states, and regions;
- a README, technical documentation, and an executive video/script of up to five
  minutes.

If this file conflicts with an explicit user instruction, follow the user's
instruction and document the impact.
