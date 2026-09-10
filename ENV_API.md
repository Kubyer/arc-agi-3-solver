# ARC-AGI-3 Environment API — Scout Report

Verified 2026-09-10 against `arc-agi 0.9.9` + `arcengine 0.9.3` (installed in `code/.venv`,
Python 3.12). Test games: `bt11`, `bt33` (in `ARC-AGI/test_environment_files/`).
The 25 public demo games were NOT downloadable from this machine (API calls to
`three.arcprize.org` time out here) — fetch them via the Kaggle competition
bundle or a networked machine into an `environment_files/` dir (see §8).

## 1. Setup

```bash
cd code/
python3 -m venv .venv
.venv/bin/pip install ./ARC-AGI        # pulls arcengine, pydantic, flask, etc.
```

Key imports:

```python
from arcengine import GameAction, GameState, FrameDataRaw
from arc_agi import Arcade, OperationMode
```

## 2. Observation format (`FrameDataRaw`)

`reset()` / `step()` return `FrameDataRaw` (pydantic model):

| field | type | meaning |
|---|---|---|
| `frame` | **runtime-only property** → `list[np.ndarray]` | 1–N frames; each `(64, 64)`, dtype **`int8`**, values 0–15 (16 colors). **NOT in `model_fields`** — it's a private attr with a property. |
| `state` | `GameState` | `NOT_PLAYED` / `NOT_FINISHED` / `WIN` / `GAME_OVER` |
| `levels_completed` | int | levels solved so far (`_score`) |
| `win_levels` | int | total levels in game |
| `available_actions` | `list[int]` | action ids legal **right now** (subset of 0–7) |
| `action_input` | `ActionInput` | echo of the action just sent (`id`, `data`, `reasoning`) |
| `guid` | str | run id for scorecard |
| `full_reset` | bool | True if the last RESET was a full game reset |
| `game_id` | str | e.g. `"bt11-fd9df0622a1a"` |

**One action can return multiple frames** (engine loops `step()` until the
action completes). In practice: use the **last** frame, or all of them.

## 3. Action space (`GameAction`)

```
RESET=0, ACTION1=1, ACTION2=2, ACTION3=3, ACTION4=4, ACTION5=5,
ACTION6=6 (COMPLEX — coordinate click, needs data={"x":0..63,"y":0..63}),
ACTION7=7
```

- Simple actions: `ACTION1–5, ACTION7` (+ `RESET`). Game-specific meaning
  (up/down/left/right/confirm…) — **undocumented, must be discovered per game**.
  `available_actions` tells you which are legal now.
- `ACTION6`: `step(GameAction.ACTION6, data={"x": int, "y": int})` — x/y in 0–63.
- Helpers: `a.is_complex()`, `a.is_simple()`, `GameAction.from_id(i)`,
  `GameAction.all_simple()`.
- ⚠️ **GOTCHA**: `action.set_data({...})` (used in the Kaggle-Starter's
  `my_agent.py`) does **NOT** reach the game through raw `arc_agi`'s
  `wrapper.step()` — it builds `ActionInput(id=action, data=data or {})` and
  ignores the enum's `action_data`. **Always pass `data=` as a kwarg to `step()`.**
  (The starter's Agents *framework* handles `set_data` itself; raw `arc_agi` does not.)

## 4. Episode loop

```python
client = Arcade(operation_mode=OperationMode.OFFLINE,
                environments_dir="ARC-AGI/test_environment_files")
env = client.make(game_id="bt11", seed=0)   # seed -> deterministic
obs = env.reset()                            # -> FrameDataRaw

while True:
    if obs.state == GameState.WIN:
        break                                # whole game solved
    if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
        obs = env.step(GameAction.RESET)     # mandatory after GAME_OVER/WIN
        continue
    # legal actions right now:
    legal = [GameAction.from_id(i) for i in obs.available_actions]
    # or: env.action_space  (same thing as GameAction list)
    a = choose(legal)                        # your policy here
    data = {"x": x, "y": y} if a is GameAction.ACTION6 else None
    obs = env.step(a, data=data, reasoning=None)
    frame = obs.frame[-1]                    # (64,64) int8, values 0..15
```

- `reset()` = sends a RESET action, returns initial obs (state usually
  `NOT_FINISHED`; `NOT_PLAYED` before first real action in some flows).
- `step(action, data=None, reasoning=None)` — `reasoning` is free-form, recorded
  only if recordings are on; it does **not** affect scoring.

## 5. State machine / termination

- `WIN` = **all** levels solved (`levels_completed == win_levels`). Game stops
  accepting actions until RESET.
- `GAME_OVER` = current level lost. Send `RESET` → level restarts
  (`full_reset=False`, score/level preserved). See RESET semantics below.
- ⚠️ **GOTCHA**: after `GAME_OVER` or `WIN`, any non-RESET action returns
  immediately with **`frame == []`** (empty list). Always branch on `state`
  **before** touching `obs.frame`.
- Level advance: winning a level increments `levels_completed` **immediately**,
  but the new level's sprites load on the **next** action. (Don't assume the
  returned frame shows the new level.)

## 6. RESET semantics (`handle_reset`)

| situation | effect |
|---|---|
| `_action_count == 0` or state == `WIN` | `full_reset()`: score=0, back to level 0, `full_reset=True` |
| otherwise | `level_reset()`: current level restarts, score kept, `full_reset=False` |

- Env var `ONLY_RESET_LEVELS=true` forces level-reset always (except on WIN).
- Scorecard counts actions 1–7 via `take_action`; **RESET (0) is never counted**
  as an action — retries are free score-wise (but cost wall-clock time).

## 7. Scoring (RHAE)

Per **completed** level:

```
level_score = min( ((baseline_actions / actions_taken) ** 2) * 100 , 115.0 )
```

- `baseline_actions`: per-level human baseline from the game's `metadata.json`
  (e.g. bt11: `[4, 8, 16, 20, 24]`). Uncompleted level → 0.
- Environment score = **level-index-weighted average**:
  `Σ(level_score[i] * i) / Σ(i)`, capped so uncompleted levels can't inflate it.
  Later levels count more.
- Only in-game actions (1–7) count toward `actions_taken`. Reasoning,
  tool calls, and RESETs are free.
- Inspect programmatically: `client.get_scorecard()` /
  `client.close_scorecard()` → `EnvironmentScorecard`
  (per-env `EnvironmentScore`: `score`, `level_scores`, `level_actions`,
  `level_baseline_actions`, `levels_completed`).

## 8. Enumerating environments & seeds

```python
envs = client.get_environments()   # OFFLINE: scans environments_dir for metadata.json
for e in envs:
    print(e.game_id, e.title, e.baseline_actions, e.default_fps)
```

- `make(game_id, seed=N)` — `game_id` may be short (`"bt11"`) or versioned
  (`"bt11-fd9df0622a1a"`); short form picks the latest downloaded version.
  Same seed ⇒ deterministic play.
- The 25 public demo games ship via the Kaggle competition / the
  `three.arcprize.org` API (`Arcade()` in `NORMAL` mode downloads on demand).
  ~~This VM can't reach that API — download `environment_files/` elsewhere and
  point `environments_dir` at it, or use `OperationMode.OFFLINE` locally.~~
  **UPDATE 2026-09-10:** all 25 real public demo games were obtained from the
  public GitHub repo `dp-web4/arc-sage` (its `environment_files/`, redistributed
  from the ARC Prize SDK) and copied to `code/environment_files/`
  (ar25, bp35, cd82, cn04, dc22, ft09, g50t, ka59, lf52, lp85, ls20, m0r0, r11l,
  re86, s5i5, sb26, sc25, sk48, sp80, su15, tn36, tr87, tu93, vc33, wa30).
  Verified: `Arcade(OperationMode.OFFLINE, environments_dir="environment_files")`
  finds and loads all 25. Use for all benchmarking.

## 9. Performance (measured)

| workload | steps/sec |
|---|---|
| random actions, bt11 (trivial game) | ~3,100 |
| random actions, bt33 (click game) | ~3,200 |
| scripted winning run, bt11 (72 steps, 5 levels → WIN) | ~1,000 |

Resets are instant. No GPU needed. Engine advertises ≥1,000 fps; real games
will be slower than these toy games but the loop overhead is negligible.

## 10. Gotchas checklist for builders

1. `obs.frame` is a **list** of `(64,64)` int8 arrays — take `frame[-1]`; check
   `state` first (empty list after GAME_OVER/WIN).
2. Pass ACTION6 coords via `step(..., data={"x":..,"y":..})` — `set_data()` is
   ignored by raw `arc_agi`.
3. `available_actions` changes per state — always read it fresh; never hardcode.
4. Action meanings (which key = up/left/…) are per-game and undiscoverable from
   the API — exploration must learn them.
5. After a level-winning action, the frame may still show the OLD level.
6. RESET is score-free; use it liberally. `GameState` enum compare with `is`.
7. `make()` returns `None` on failure (game not found) — check it.
8. Recordings: `make(..., save_recording=True)` writes JSONL per run (can be
   large with `include_frame_data=True`).
9. `FrameDataRaw.frame` won't survive `model_dump()`/JSON — it's runtime-only.
10. Network: `NORMAL`/`ONLINE` modes need `three.arcprize.org`; anonymous key
    auto-fetch also needs network. Default to `OFFLINE` for dev.

## 11. Reference files in this repo

- `probe_api.py` — API surface dump (actions, states, frame shapes)
- `verify_loop.py` — random agent + scripted bt11 solver (runnable proof)
- `.venv/` — Python 3.12 venv with `arc-agi 0.9.9`, `arcengine 0.9.3`
- `ARC-AGI/` — toolkit source (cloned 2026-09-10)
- `ARC-AGI-3-Kaggle-Starter/` — official starter (agent framework pattern,
  `make submit` packaging)
