# ROCmHub Product UX Specification

## Home

- **User goal:** Understand ROCmHub in seconds and begin a model-to-artifact workflow.
- **Primary action:** Explore a model.
- **Secondary action:** Open Forge Studio.
- **Progress state:** Recent work shows compact, human-readable stage and progress context.
- **Result state:** Completed work appears as a useful continuation point, not a data table.
- **Advanced details:** Service health and runtime diagnostics remain available through the global status control.

## Model Explorer

- **User goal:** Identify a model and understand whether it is ready to prepare for an AMD target.
- **Primary action:** Inspect a model, then continue to Forge.
- **Secondary action:** Choose a curated model starting point.
- **Progress state:** The discovery canvas becomes an inspection state with a deliberate loading transition.
- **Result state:** A full-width model profile summarizes identity, architecture, compatibility, readiness, and next step.
- **Advanced details:** Revision, source metadata, raw identifiers, and inspection payload live in an expandable details area.

## Forge Studio

- **User goal:** Turn a selected model into a reproducible, target-aware build artifact.
- **Primary action:** Advance through model, target, profile, review, and build stages.
- **Secondary action:** Revisit an earlier stage without losing the current plan.
- **Progress state:** A stage rail and focused workspace show exactly what is being prepared or built.
- **Result state:** Artifact-first completion view with outcome, build summary, and clear next actions.
- **Advanced details:** Overrides, raw plan, job identifiers, logs, digests, and environment data are disclosed secondarily.

## AI Engineer

- **User goal:** State an infrastructure objective and receive an actionable expert recommendation.
- **Primary action:** Run the selected objective for a chosen model.
- **Secondary action:** Start from a suggested objective or refine the model and target.
- **Progress state:** A restrained activity sequence communicates inspection, environment checks, planning, evaluation, and finalization without exposing hidden reasoning.
- **Result state:** Executive recommendation and actions appear before supporting trajectory and evidence.
- **Advanced details:** Provider configuration, budgets, job metadata, and technical trajectory are collapsed by default.

## Optimization Lab

- **User goal:** Compare a baseline with candidate configurations and decide what to validate next.
- **Primary action:** Run the comparison.
- **Secondary action:** Adjust candidate strategy or inspect measurement availability.
- **Progress state:** Baseline and candidate lanes remain visible while the experiment is queued or running.
- **Result state:** A comparison-first result explains measured outcomes or, on non-ROCm hosts, the useful plan that is ready for hardware validation.
- **Advanced details:** Recipes, raw metrics, job identifiers, and environment diagnostics are expandable.

## Global navigation and status

- **User goal:** Move between the five product stages while retaining context.
- **Primary action:** Navigate through Home, Models, Forge, Engineer, and Optimize.
- **Secondary action:** Inspect service and host status.
- **Progress state:** Current location and active jobs are legible without dominating the interface.
- **Result state:** Cross-workflow handoffs preserve the selected model or job.
- **Advanced details:** Backend version, runtime, storage, and low-level health fields live in the status disclosure.
