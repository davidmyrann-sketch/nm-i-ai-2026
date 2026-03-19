#!/usr/bin/env python3
"""
NM i AI 2026 — Astar Island Norse World Prediction
Usage: python predict.py --token YOUR_JWT_TOKEN [--submit]
"""
import os, sys, json, time, argparse
import numpy as np
import requests

BASE = "https://api.ainm.no"
N_CLASSES = 6

# Terrain codes in grid
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5


def terrain_to_class(t):
    if t in (OCEAN, PLAINS, EMPTY): return 0
    if t == SETTLEMENT: return 1
    if t == PORT: return 2
    if t == RUIN: return 3
    if t == FOREST: return 4
    if t == MOUNTAIN: return 5
    return 0


def is_static(t):
    return t in (MOUNTAIN, OCEAN)


def get_prior(initial_terrain, near_settlement=False, near_coast=False):
    """
    Prior probability distribution based on initial terrain.
    Higher settlement probability for plains near existing settlements.
    """
    p = np.zeros(N_CLASSES)
    if initial_terrain in (OCEAN, EMPTY):
        p[0] = 1.0
    elif initial_terrain == MOUNTAIN:
        p[5] = 1.0
    elif initial_terrain == PLAINS:
        if near_settlement and near_coast:
            p[0] = 0.45; p[1] = 0.20; p[2] = 0.15; p[3] = 0.10; p[4] = 0.07; p[5] = 0.03
        elif near_settlement:
            p[0] = 0.50; p[1] = 0.25; p[2] = 0.05; p[3] = 0.10; p[4] = 0.08; p[5] = 0.02
        else:
            p[0] = 0.65; p[1] = 0.10; p[2] = 0.03; p[3] = 0.08; p[4] = 0.12; p[5] = 0.02
    elif initial_terrain == FOREST:
        p[4] = 0.75; p[0] = 0.08; p[1] = 0.06; p[2] = 0.02; p[3] = 0.07; p[5] = 0.02
    elif initial_terrain == SETTLEMENT:
        p[1] = 0.45; p[2] = 0.20; p[3] = 0.25; p[0] = 0.07; p[4] = 0.02; p[5] = 0.01
    elif initial_terrain == PORT:
        p[2] = 0.50; p[1] = 0.10; p[3] = 0.28; p[0] = 0.09; p[4] = 0.02; p[5] = 0.01
    elif initial_terrain == RUIN:
        p[3] = 0.35; p[0] = 0.25; p[1] = 0.20; p[4] = 0.12; p[2] = 0.05; p[5] = 0.03
    else:
        p[0] = 1.0
    return p / p.sum()


def apply_floor(pred, floor=0.01):
    pred = np.maximum(pred, floor)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


class AstarPredictor:
    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def get_rounds(self):
        return self.session.get(f"{BASE}/astar-island/rounds").json()

    def get_round(self, round_id):
        return self.session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

    def get_budget(self):
        return self.session.get(f"{BASE}/astar-island/budget").json()

    def simulate(self, round_id, seed_idx, vx, vy, vw=15, vh=15):
        r = self.session.post(f"{BASE}/astar-island/simulate", json={
            "round_id": round_id, "seed_index": seed_idx,
            "viewport_x": vx, "viewport_y": vy,
            "viewport_w": vw, "viewport_h": vh
        })
        return r.json()

    def submit(self, round_id, seed_idx, prediction):
        r = self.session.post(f"{BASE}/astar-island/submit", json={
            "round_id": round_id, "seed_index": seed_idx,
            "prediction": prediction
        })
        return r.json()

    def build_prediction(self, H, W, initial_grid, initial_settlements, obs_counts, obs_total):
        """Build H×W×6 prediction tensor combining priors and observations."""
        # Find initial settlement positions for proximity features
        settlement_positions = set()
        coast_positions = set()
        for s in initial_settlements:
            settlement_positions.add((s["x"], s["y"]))

        # Determine coastal cells (adjacent to ocean)
        for y in range(H):
            for x in range(W):
                if initial_grid[y][x] in (OCEAN,):
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                coast_positions.add((nx, ny))

        prediction = np.zeros((H, W, N_CLASSES))

        for y in range(H):
            for x in range(W):
                initial_t = initial_grid[y][x]

                if obs_total[y][x] > 0:
                    # Use empirical distribution from observations
                    probs = obs_counts[y][x] / obs_total[y][x]
                    # Override static cells
                    if is_static(initial_t):
                        cls = terrain_to_class(initial_t)
                        probs = np.zeros(N_CLASSES)
                        probs[cls] = 1.0
                else:
                    # Use prior based on initial terrain and context
                    near_s = any(abs(sx - x) + abs(sy - y) <= 5 for sx, sy in settlement_positions)
                    near_c = (x, y) in coast_positions
                    probs = get_prior(initial_t, near_s, near_c)

                prediction[y][x] = probs

        # Apply floor to avoid zero probabilities (would destroy KL divergence score)
        prediction = apply_floor(prediction, floor=0.01)
        return prediction

    def run(self, dry_run=False):
        # Get active round
        rounds = self.get_rounds()
        active = [r for r in rounds if r["status"] == "active"]
        if not active:
            print("No active round! Checking all rounds...")
            print(json.dumps(rounds, indent=2))
            return

        round_info = active[0]
        round_id = round_info["id"]
        W = round_info["map_width"]
        H = round_info["map_height"]
        print(f"Round {round_info['round_number']}: {W}×{H}, closes {round_info.get('closes_at', 'unknown')}")

        # Get full round details
        details = self.get_round(round_id)
        initial_states = details["initial_states"]
        n_seeds = len(initial_states)
        print(f"Seeds: {n_seeds}")

        budget = self.get_budget()
        queries_used = budget.get("queries_used", 0)
        queries_max = budget.get("queries_max", 50)
        queries_left = queries_max - queries_used
        print(f"Budget: {queries_used}/{queries_max} used, {queries_left} remaining")

        if queries_left == 0:
            print("Budget exhausted — using priors only")

        # Build viewport tiles to cover the 40×40 map with 15×15 windows
        # Using step=13 to get slight overlap at edges
        vp_positions = []
        step = 13
        for vy in range(0, H, step):
            for vx in range(0, W, step):
                vx_clamped = min(vx, W - 15)
                vy_clamped = min(vy, H - 15)
                vp_positions.append((vx_clamped, vy_clamped))
        # Deduplicate while preserving order
        seen = set()
        vp_positions = [p for p in vp_positions if not (p in seen or seen.add(p))]
        print(f"Viewport tiles: {len(vp_positions)} (covers {W}×{H} map)")

        # Allocate queries across seeds
        queries_per_seed = max(0, queries_left // n_seeds)
        extra_queries = max(0, queries_left - queries_per_seed * n_seeds)
        print(f"Queries per seed: {queries_per_seed} (+{extra_queries} extra for seed 0)")

        for seed_idx in range(n_seeds):
            print(f"\n--- Seed {seed_idx}/{n_seeds - 1} ---")
            initial_grid = initial_states[seed_idx]["grid"]
            initial_settlements = initial_states[seed_idx].get("settlements", [])

            obs_counts = np.zeros((H, W, N_CLASSES))
            obs_total = np.zeros((H, W))

            # Number of queries to use for this seed
            n_queries = queries_per_seed + (extra_queries if seed_idx == 0 else 0)
            tiles_to_query = vp_positions[:n_queries]

            for i, (vx, vy) in enumerate(tiles_to_query):
                print(f"  Query {i+1}/{len(tiles_to_query)}: viewport ({vx},{vy},15×15) ...", end=" ", flush=True)
                try:
                    result = self.simulate(round_id, seed_idx, vx, vy, 15, 15)
                    if "grid" not in result:
                        print(f"ERROR: {result}")
                        time.sleep(1)
                        continue

                    grid = result["grid"]
                    actual_vp = result.get("viewport", {"x": vx, "y": vy})
                    ax, ay = actual_vp["x"], actual_vp["y"]

                    for row_i, row in enumerate(grid):
                        for col_i, terrain in enumerate(row):
                            world_y = ay + row_i
                            world_x = ax + col_i
                            if 0 <= world_y < H and 0 <= world_x < W:
                                cls = terrain_to_class(terrain)
                                obs_counts[world_y][world_x][cls] += 1
                                obs_total[world_y][world_x] += 1

                    queries_so_far = result.get("queries_used", "?")
                    print(f"ok (budget used: {queries_so_far})")
                    time.sleep(0.25)  # Respect 5 req/s rate limit

                except Exception as e:
                    print(f"EXCEPTION: {e}")
                    time.sleep(2)

            # Build prediction tensor
            cells_observed = int((obs_total > 0).sum())
            total_cells = H * W
            print(f"  Observed {cells_observed}/{total_cells} cells ({100*cells_observed/total_cells:.1f}%)")

            prediction = self.build_prediction(H, W, initial_grid, initial_settlements, obs_counts, obs_total)

            # Verify all rows sum to 1.0
            row_sums = prediction.sum(axis=-1)
            assert np.allclose(row_sums, 1.0, atol=0.01), f"Prediction rows don't sum to 1: min={row_sums.min():.4f}"

            if dry_run:
                print(f"  [DRY RUN] Would submit prediction for seed {seed_idx}")
                print(f"  Sample cell (0,0): {prediction[0][0].tolist()}")
                continue

            print(f"  Submitting prediction for seed {seed_idx}...", end=" ", flush=True)
            result = self.submit(round_id, seed_idx, prediction.tolist())
            print(f"  {result}")
            time.sleep(0.6)  # Respect 2 req/s rate limit

        print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Astar Island prediction for NM i AI 2026")
    parser.add_argument("--token", default=os.environ.get("AINM_TOKEN", ""), help="JWT token from app.ainm.no")
    parser.add_argument("--dry-run", action="store_true", help="Query simulator but don't submit predictions")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: No token provided. Use --token YOUR_JWT or set AINM_TOKEN env var.")
        print("Get your token from app.ainm.no (browser cookies: access_token)")
        sys.exit(1)

    predictor = AstarPredictor(args.token)
    predictor.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
