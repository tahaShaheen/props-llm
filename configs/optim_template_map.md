# Config to optimizer template map

This map excludes env-description templates and focuses only on optimizer prompt templates.

## Grouped (at a glance)

- `num_optim.j2`
  - `configs/cartpole/cartpole_props.yaml`
  - `configs/hopper/hopper_linear_policy_props.yaml`
  - `configs/inverted_double_pendulum/inverteddoublependulum_props.yaml`
  - `configs/invertedpendulum/invertedpendulum_props.yaml`
  - `configs/mountaincar/mountaincar_props.yaml`
  - `configs/mountaincarcontinuous/mountaincar_continuous_props.yaml`
  - `configs/nav/nav_props.yaml`
  - `configs/pong/pong_props.yaml`
  - `configs/reacher/reacher_props.yaml`
  - `configs/swimmer/swimmer_props.yaml`
  - `configs/walker2d/walker2d_props.yaml`

- `num_optim_semantic.j2`
  - `configs/cartpole/cartpole_propsp.yaml`
  - `configs/hopper/hopper_linear_policy_propsp.yaml`
  - `configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml`
  - `configs/invertedpendulum/invertedpendulum_propsp.yaml`
  - `configs/mountaincar/mountaincar_propsp.yaml`
  - `configs/mountaincarcontinuous/mountaincar_continuous_propsp.yaml`
  - `configs/nav/nav_propsp.yaml`
  - `configs/pong/pong_propsp.yaml`
  - `configs/reacher/reacher_propsp.yaml`
  - `configs/swimmer/swimmer_propsp.yaml`
  - `configs/walker2d/walker2d_propsp.yaml`

- `num_optim_candidates.j2`
  - `configs/cliffwalking/cliffwalking_props.yaml`
  - `configs/frozenlake/frozenlake_props.yaml`
  - `configs/maze/maze_props.yaml`
  - `configs/nim/nim_props.yaml`

- `num_optim_candidates_semantics.j2`
  - `configs/cliffwalking/cliffwalking_propsp.yaml`
  - `configs/frozenlake/frozenlake_propsp.yaml`
  - `configs/maze/maze_propsp.yaml`
  - `configs/nim/nim_propsp.yaml`

- `num_optim_semantic_feedback.j2`
  - `configs/walker2d/walker2d_propspf.yaml`

- `num_optim_candidates_semantics_feedback.j2`
  - `configs/frozenlake/frozenlake_propspf.yaml`

## Full mapping (per config)

| Config YAML | Task | llm_si_template_name | llm_output_conversion_template_name |
|---|---|---|---|
| `configs/cartpole/cartpole_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/cartpole/cartpole_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/cliffwalking/cliffwalking_props.yaml` | `dist_state_llm_num_optim` | `num_optim_candidates.j2` | `num_optim_candidates.j2` |
| `configs/cliffwalking/cliffwalking_propsp.yaml` | `dist_state_llm_num_optim_semantics` | `num_optim_candidates_semantics.j2` | `num_optim_candidates_semantics.j2` |
| `configs/frozenlake/frozenlake_props.yaml` | `dist_state_llm_num_optim` | `num_optim_candidates.j2` | `num_optim_candidates.j2` |
| `configs/frozenlake/frozenlake_propsp.yaml` | `dist_state_llm_num_optim_semantics` | `num_optim_candidates_semantics.j2` | `num_optim_candidates_semantics.j2` |
| `configs/frozenlake/frozenlake_propspf.yaml` | `dist_state_llm_num_optim_semantics_with_feedback` | `num_optim_candidates_semantics_feedback.j2` | `num_optim_candidates_semantics_feedback.j2` |
| `configs/hopper/hopper_linear_policy_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/hopper/hopper_linear_policy_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/inverted_double_pendulum/inverteddoublependulum_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/invertedpendulum/invertedpendulum_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/invertedpendulum/invertedpendulum_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/maze/maze_props.yaml` | `dist_state_llm_num_optim` | `num_optim_candidates.j2` | `num_optim_candidates.j2` |
| `configs/maze/maze_propsp.yaml` | `dist_state_llm_num_optim_semantics` | `num_optim_candidates_semantics.j2` | `num_optim_candidates_semantics.j2` |
| `configs/mountaincar/mountaincar_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/mountaincar/mountaincar_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/mountaincarcontinuous/mountaincar_continuous_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/mountaincarcontinuous/mountaincar_continuous_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/nav/nav_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/nav/nav_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/nim/nim_props.yaml` | `dist_state_llm_num_optim` | `num_optim_candidates.j2` | `num_optim_candidates.j2` |
| `configs/nim/nim_propsp.yaml` | `dist_state_llm_num_optim_semantics` | `num_optim_candidates_semantics.j2` | `num_optim_candidates_semantics.j2` |
| `configs/pong/pong_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/pong/pong_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/reacher/reacher_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/reacher/reacher_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/swimmer/swimmer_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/swimmer/swimmer_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/walker2d/walker2d_props.yaml` | `cont_space_llm_num_optim` | `num_optim.j2` | `num_optim.j2` |
| `configs/walker2d/walker2d_propsp.yaml` | `cont_state_llm_num_optim_semantics` | `num_optim_semantic.j2` | `num_optim_semantic.j2` |
| `configs/walker2d/walker2d_propspf.yaml` | `cont_state_llm_num_optim_semantics_with_feedback` | `num_optim_semantic_feedback.j2` | `num_optim_semantic_feedback.j2` |
