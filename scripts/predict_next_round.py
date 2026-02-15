from euroleague_sim.cli import main

if __name__ == "__main__":
    # Equivalent to: euroleague-sim predict --season 2025 --round next --n-sims 50000
    main(["predict", "--season", "2025", "--round", "next", "--n-sims", "50000"])
