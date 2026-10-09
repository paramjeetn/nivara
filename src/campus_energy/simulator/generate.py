"""CLI entry point for data generation."""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from campus_energy.config import load_config
from campus_energy.simulator.campus import simulate_campus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic campus energy data")
    parser.add_argument("--config", type=str, help="Path to simulation.yaml")
    parser.add_argument("--output-dir", type=str, default="data/generated", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting simulation for campus: {config.campus.name}")
    logger.info(f"Period: {config.simulation.start_date} to {config.simulation.end_date}")
    
    datasets = simulate_campus(config)
    
    assets_df = datasets["assets"]
    telemetry_df = datasets["telemetry"]
    context_df = datasets["context"]
    
    # Inject events
    from campus_energy.simulator.events import inject_events
    tel_df, events_df, gt_df = inject_events(telemetry_df, context_df, assets_df, config.simulation.seed)
    
    logger.info(f"Generated {len(assets_df)} assets")
    logger.info(f"Generated {len(tel_df)} telemetry readings (after injection)")
    logger.info(f"Generated {len(context_df)} context records")
    logger.info(f"Injected {len(events_df)} anomalous events")
    
    # Save to CSV
    assets_path = output_dir / "assets.csv"
    telemetry_path = output_dir / "telemetry.csv"
    context_path = output_dir / "context.csv"
    events_path = output_dir / "events.csv"
    gt_path = output_dir / "anomaly_ground_truth.csv"
    complaints_path = output_dir / "complaints.csv"
    
    assets_df.to_csv(assets_path, index=False)
    tel_df.to_csv(telemetry_path, index=False)
    context_df.to_csv(context_path, index=False)
    events_df.to_csv(events_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    
    # Generate correlated complaints
    from campus_energy.simulator.complaints import generate_complaints
    complaints_df = generate_complaints(events_df, assets_df, config.simulation.seed)
    complaints_df.to_csv(complaints_path, index=False)
    logger.info(f"Generated {len(complaints_df)} NLP complaints")
    
    logger.info("Saved all CSV files")
    
    # Generate manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "config_version": config.simulation.version,
        "seed": config.simulation.seed,
        "campus_name": config.campus.name,
        "start_date": config.simulation.start_date,
        "end_date": config.simulation.end_date,
        "interval_minutes": config.simulation.interval_minutes,
        "records": {
            "assets": len(assets_df),
            "telemetry": len(telemetry_df),
            "context": len(context_df),
        }
    }
    
    with open(manifest_dir / "simulation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info("Simulation complete.")

if __name__ == "__main__":
    main()
