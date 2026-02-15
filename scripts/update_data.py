from euroleague_sim.cli import main

if __name__ == "__main__":
    # Equivalent to: euroleague-sim update-data --season 2025 --history 2
    main(["update-data", "--season", "2025", "--history", "2"])
